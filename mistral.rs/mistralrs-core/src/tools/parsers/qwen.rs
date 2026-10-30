//! Qwen tool call parser.
//!
//! Formats:
//! `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`
//! `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
//! `<tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value></tool_call>` (GLM style; Spark's
//! own template). Without it, Spark's calls only parsed because the grammar forced them into one of the
//! Qwen forms, which its training never showed it.

use candle_core::Result;
use llguidance::api::TopLevelGrammar;
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_OPEN: &str = "<parameter=";
const PARAMETER_CLOSE: &str = "</parameter>";
const ARG_KEY_OPEN: &str = "<arg_key>";
const ARG_KEY_CLOSE: &str = "</arg_key>";
const ARG_VALUE_OPEN: &str = "<arg_value>";
const ARG_VALUE_CLOSE: &str = "</arg_value>";

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, false),
            tools,
            "arguments",
            false,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            qwen_tool_call_lark(tools, true),
            tools,
            "arguments",
            false,
        )
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        let re = QWEN_REGEX
            .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

        if !re.is_match(message) {
            Ok(None)
        } else {
            parse_qwen_tool_calls(message)
        }
    }
}

fn qwen_tool_call_lark(tools: &[Tool], include_wrapper: bool) -> String {
    // Parallel calls arrive as consecutive <tool_call> blocks, so the grammar stays open for more
    let start = if include_wrapper {
        r#"start: "<tool_call>" (json_call | xml_call | glm_call) ("\n"? "<tool_call>" (json_call | xml_call | glm_call))*"#
    } else {
        r#"start: (json_call | xml_call | glm_call) ("\n"? <tool_call> (json_call | xml_call | glm_call))*"#
    };
    let json_call = if include_wrapper {
        r#"json_call: @json_body "</tool_call>""#
    } else {
        "json_call: @json_body </tool_call>"
    };
    let xml_end = if include_wrapper {
        r#""</tool_call>""#
    } else {
        "</tool_call>"
    };

    format!(
        r#"{start}
{json_call}
xml_call: "\n"? xml_function "\n"? {xml_end}
{}
xml_param_value: (xml_param_text | xml_param_lt)*
xml_param_text: /[^<]+/
xml_param_lt: {}
{}
glm_call: glm_function {xml_end}
{}
glm_value: (xml_param_text | glm_value_lt)*
glm_value_lt: {}"#,
        qwen_xml_function_rules(tools),
        xml_param_lt_rule(),
        qwen_xml_generic_rules(tools),
        glm_function_rules(tools),
        lt_rule("/arg_value>"),
    )
}

// Any `<` that does not open the closing `</parameter>` tag may appear inside a value (HTML, XML, code).
// Once a value holds `</parameter`, the only continuation is `>`: nothing a value carries looks like that
// except a botched close. Allowing `</parameter=path>` as value text let Spark-4B write it, then
// `</function>` and `</invoke>` until max_tokens, all inside the value, so the call never closed or parsed.
fn xml_param_lt_rule() -> String {
    lt_rule("/parameter>")
}

/// `<` inside a value that ends at `<CLOSE`: anything but the close tag, and a started close tag can only
/// finish (see xml_param_lt_rule).
fn lt_rule(close: &str) -> String {
    let mut alternatives = vec![r#""<" /[^\/]/"#.to_string()];
    for (idx, ch) in close.chars().enumerate().skip(1).filter(|(_, ch)| *ch != '>') {
        let prefix = &close[..idx];
        let escaped = if ch == '/' {
            "\\/".to_string()
        } else {
            ch.to_string()
        };
        alternatives.push(format!(r#""<{prefix}" /[^{escaped}]/"#));
    }
    alternatives.join(" | ")
}

#[derive(serde::Serialize)]
struct QwenToolCall {
    name: String,
    arguments: Value,
}

fn parse_qwen_tool_calls(message: &str) -> Result<Option<String>> {
    let re = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    let mut xml_calls = Vec::new();
    let mut json_calls = Vec::new();

    for caps in re.captures_iter(message) {
        let inner = caps.name("inner").unwrap().as_str().trim();
        if inner.is_empty() {
            continue;
        }

        if inner.starts_with(FUNCTION_OPEN) {
            match parse_qwen_xml_tool_call(inner) {
                Some(call) => xml_calls.push(call),
                None => return Ok(None),
            }
            continue;
        }

        if !inner.starts_with('{') {
            match parse_glm_tool_call(inner) {
                Some(call) => xml_calls.push(call),
                None => return Ok(None),
            }
            continue;
        }

        match serde_json::from_str::<Value>(inner) {
            Ok(value) => json_calls.push(value),
            Err(_) => return Ok(None),
        }
    }

    if !xml_calls.is_empty() {
        return Ok(Some(
            serde_json::to_string(&xml_calls).map_err(candle_core::Error::msg)?,
        ));
    }

    match json_calls.len() {
        0 => Ok(None),
        1 => Ok(Some(
            serde_json::to_string(&json_calls[0]).map_err(candle_core::Error::msg)?,
        )),
        _ => Ok(Some(
            serde_json::to_string(&json_calls).map_err(candle_core::Error::msg)?,
        )),
    }
}

// One function per block: text quoted inside a value must not be able to open a second call.
fn parse_qwen_xml_tool_call(inner: &str) -> Option<QwenToolCall> {
    let (name, rest) = inner.strip_prefix(FUNCTION_OPEN)?.split_once('>')?;
    let body = rest.trim_end().strip_suffix(FUNCTION_CLOSE)?;
    if name.contains('\n') {
        return None;
    }

    let mut arguments = Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAMETER_OPEN) {
        let (key, value_and_rest) = rest[start + PARAMETER_OPEN.len()..].split_once('>')?;
        let (value, after) = split_parameter_value(value_and_rest)?;
        if key.contains('\n')
            || arguments
                .insert(key.trim().to_string(), qwen_xml_param_value(value))
                .is_some()
        {
            return None;
        }
        rest = after;
    }
    Some(QwenToolCall {
        name: name.trim().to_string(),
        arguments: Value::Object(arguments),
    })
}

// `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`: values are kept verbatim (the template writes a string
// as-is and anything else as JSON); types come from the tool schema later, as for the XML form.
fn parse_glm_tool_call(inner: &str) -> Option<QwenToolCall> {
    let (name, mut rest) = match inner.find(ARG_KEY_OPEN) {
        Some(i) => (inner[..i].trim(), &inner[i..]),
        None => (inner.trim(), ""),
    };
    if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c == '<' || c == '{') {
        return None;
    }
    let mut arguments = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let after_key = rest.strip_prefix(ARG_KEY_OPEN)?;
        let (key, after) = after_key.split_once(ARG_KEY_CLOSE)?;
        let after = after.trim_start().strip_prefix(ARG_VALUE_OPEN)?;
        // a value ends at a close tag followed by the next key or the end of the call
        let (value, next) = after.match_indices(ARG_VALUE_CLOSE).find_map(|(idx, _)| {
            let next = &after[idx + ARG_VALUE_CLOSE.len()..];
            let t = next.trim_start();
            (t.is_empty() || t.starts_with(ARG_KEY_OPEN)).then_some((&after[..idx], next))
        })?;
        if arguments.insert(key.trim().to_string(), Value::String(value.to_string())).is_some() {
            return None;
        }
        rest = next;
    }
    Some(QwenToolCall { name: name.to_string(), arguments: Value::Object(arguments) })
}

// A value ends only at a close tag followed by the next parameter or the end of the function body.
fn split_parameter_value(text: &str) -> Option<(&str, &str)> {
    text.match_indices(PARAMETER_CLOSE).find_map(|(idx, _)| {
        let after = &text[idx + PARAMETER_CLOSE.len()..];
        let next = after.trim_start();
        (next.is_empty() || next.starts_with(PARAMETER_OPEN)).then_some((&text[..idx], after))
    })
}

// Values are kept verbatim minus the template's single framing newlines; types come from the tool schema later
fn qwen_xml_param_value(raw: &str) -> Value {
    let value = raw.strip_prefix('\n').unwrap_or(raw);
    let value = value.strip_suffix('\n').unwrap_or(value);
    Value::String(value.to_string())
}

fn qwen_xml_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();

    for (tool_idx, tool) in tools.iter().enumerate() {
        let branch = format!("qwen_xml_tool_{tool_idx}");
        let args = qwen_xml_args_rule(tool_idx, tool, &mut rules);
        // the tag prefix is its own literal: in a tokenizer where `<function=` is a special token (LFM3),
        // specialization turns it into that token; spelled as text inside one literal, the model had to
        // write it as text and the special-text guard defused it, so the call never parsed
        let opener = format!(
            "{} {} {}",
            lark_string(FUNCTION_OPEN),
            lark_string(&tool.function.name),
            lark_string(">")
        );
        rules.push(format!(
            "{branch}: {opener} \"\\n\"? {args} \"</function>\""
        ));
        branches.push(branch);
    }

    if branches.is_empty() {
        rules.push("xml_function: qwen_xml_generic_function".to_string());
    } else {
        rules.push(format!("xml_function: {}", branches.join(" | ")));
    }

    rules.join("\n")
}

fn qwen_xml_args_rule(tool_idx: usize, tool: &Tool, rules: &mut Vec<String>) -> String {
    let args_rule = format!("qwen_xml_args_{tool_idx}");
    let Some(parameters) = tool.function.parameters.as_ref() else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };
    let Some(Value::Object(properties)) = parameters.get("properties") else {
        rules.push(format!("{args_rule}: qwen_xml_generic_params"));
        return args_rule;
    };

    let required = parameters
        .get("required")
        .and_then(|v| v.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<std::collections::BTreeSet<_>>()
        })
        .unwrap_or_default();

    let mut required_pairs = Vec::new();
    let mut optional_pairs = Vec::new();
    let mut property_names = properties.keys().cloned().collect::<Vec<_>>();
    property_names.sort();

    for (prop_idx, name) in property_names.iter().enumerate() {
        let pair = format!("qwen_xml_arg_{tool_idx}_{prop_idx}");
        let opener = format!("{} {} {}", lark_string(PARAMETER_OPEN), lark_string(name), lark_string(">"));
        rules.push(format!(
            "{pair}: {opener} \"\\n\"? xml_param_value \"</parameter>\" \"\\n\"?"
        ));
        if required.contains(name.as_str()) {
            required_pairs.push(pair);
        } else {
            optional_pairs.push(pair);
        }
    }

    // Qwen does not always emit parameters in schema order, so accept any order (llama.cpp permutes too)
    let mut parts = required_pairs;
    parts.extend(optional_pairs);

    if parts.is_empty() {
        rules.push(format!("{args_rule}:"));
    } else {
        rules.push(format!("{args_rule}: ({})*", parts.join(" | ")));
    }

    args_rule
}

fn qwen_xml_generic_rules(tools: &[Tool]) -> &'static str {
    if tools.iter().any(|tool| tool.function.parameters.is_none()) {
        r#"qwen_xml_generic_function: "<function=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? qwen_xml_generic_params "</function>"
qwen_xml_generic_params: (qwen_xml_generic_param "\n"?)*
qwen_xml_generic_param: "<parameter=" /[a-zA-Z_][a-zA-Z0-9_]*/ ">" "\n"? xml_param_value "</parameter>""#
    } else {
        ""
    }
}

/// `NAME(<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>)*` per tool, keys from its schema in any order.
fn glm_function_rules(tools: &[Tool]) -> String {
    let mut rules = Vec::new();
    let mut branches = Vec::new();
    for (tool_idx, tool) in tools.iter().enumerate() {
        let keys: Vec<String> = tool
            .function
            .parameters
            .as_ref()
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
            .map(|o| {
                let mut k: Vec<String> = o.keys().cloned().collect();
                k.sort();
                k
            })
            .unwrap_or_default();
        let pairs: Vec<String> = keys
            .iter()
            .enumerate()
            .map(|(key_idx, key)| {
                let pair = format!("glm_arg_{tool_idx}_{key_idx}");
                rules.push(format!(
                    "{pair}: \"\\n\"? {} {} {} {} glm_value {}",
                    lark_string(ARG_KEY_OPEN),
                    lark_string(key),
                    lark_string(ARG_KEY_CLOSE),
                    lark_string(ARG_VALUE_OPEN),
                    lark_string(ARG_VALUE_CLOSE)
                ));
                pair
            })
            .collect();
        let branch = format!("glm_tool_{tool_idx}");
        let args = if pairs.is_empty() { String::new() } else { format!("({})*", pairs.join(" | ")) };
        rules.push(format!("{branch}: {} {args} \"\\n\"?", lark_string(&tool.function.name)));
        branches.push(branch);
    }
    if branches.is_empty() {
        // no declared tools: any identifier, any keys
        rules.push(format!(
            "glm_function: /[a-zA-Z_][a-zA-Z0-9_.-]*/ (\"\\n\"? {} /[a-zA-Z_][a-zA-Z0-9_.-]*/ {} {} glm_value {})*",
            lark_string(ARG_KEY_OPEN),
            lark_string(ARG_KEY_CLOSE),
            lark_string(ARG_VALUE_OPEN),
            lark_string(ARG_VALUE_CLOSE)
        ));
    } else {
        rules.push(format!("glm_function: {}", branches.join(" | ")));
    }
    rules.join("\n")
}

fn lark_string(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n");
    format!("\"{escaped}\"")
}

#[cfg(test)]
mod tests {
    use super::{parse_qwen_tool_calls, QwenParser};
    use crate::tools::parsers::{specialize_required_tool_call_grammar, ToolFormatParser};
    use mistralrs_mcp::{Function, ToolType};
    use serde_json::json;
    use serde_json::Value;
    use std::collections::HashMap;
    use std::sync::Arc;

    struct GreedyTokenizerEnv {
        trie: toktrie::TokTrie,
    }

    impl toktrie::TokenizerEnv for GreedyTokenizerEnv {
        fn tok_trie(&self) -> &toktrie::TokTrie {
            &self.trie
        }

        fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<toktrie::TokenId> {
            self.trie.greedy_tokenize(bytes)
        }

        fn tokenize_is_canonical(&self) -> bool {
            false
        }
    }

    fn wrapper_token_trie() -> toktrie::TokTrie {
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        tokens.push(b"\xff<eos>".to_vec());
        tokens.push(b"\xff<tool_call>".to_vec());
        tokens.push(b"\xff</tool_call>".to_vec());
        let vocab_size = u32::try_from(tokens.len()).expect("test vocabulary fits in u32");
        toktrie::TokTrie::from(&toktrie::TokRxInfo::new(vocab_size, eos), &tokens)
    }

    #[test]
    fn parses_qwen_json_tool_call() {
        let parsed = QwenParser
            .parse(r#"<tool_call>{"name":"get_weather","arguments":{"city":"Paris"}}</tool_call>"#)
            .unwrap()
            .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value["name"], "get_weather");
        assert_eq!(value["arguments"]["city"], "Paris");
    }

    #[test]
    fn parses_qwen_xml_tool_call() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=locations>
[{"country":"France","city":"Paris"}]
</parameter>
<parameter=temp_units>
celsius
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "get_weather");
        // Raw text: the matcher applies the tool schema types afterwards
        assert_eq!(
            value[0]["arguments"]["locations"],
            r#"[{"country":"France","city":"Paris"}]"#
        );
        assert_eq!(value[0]["arguments"]["temp_units"], "celsius");
    }

    #[test]
    fn qwen_xml_values_keep_inner_whitespace() {
        let parsed = parse_qwen_tool_calls(
            "<tool_call>\n<function=write_file>\n<parameter=content>\n    indented\n\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap()
        .unwrap();
        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["arguments"]["content"], "    indented\n");
    }

    #[test]
    fn parses_multiple_qwen_xml_tool_calls() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=get_weather>
<parameter=city>Tokyo</parameter>
</function>
</tool_call><tool_call>
<function=get_time>
<parameter=timezone>Asia/Tokyo</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value.as_array().unwrap().len(), 2);
        assert_eq!(value[0]["arguments"]["city"], "Tokyo");
        assert_eq!(value[1]["arguments"]["timezone"], "Asia/Tokyo");
    }

    #[test]
    fn parses_qwen_xml_code_with_less_than() {
        let parsed = parse_qwen_tool_calls(
            r#"<tool_call>
<function=mistralrs_execute_python>
<parameter=code>
print(1 < 2)
</parameter>
</function>
</tool_call>"#,
        )
        .unwrap()
        .unwrap();

        let value: Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(value[0]["name"], "mistralrs_execute_python");
        assert_eq!(value[0]["arguments"]["code"], "print(1 < 2)");
    }

    #[test]
    fn qwen_xml_grammar_requires_code_without_forcing_newline_before_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": {
                "code": { "type": "string" },
                "outputs": { "type": "array" }
            },
            "required": ["code"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "mistralrs_execute_python".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: Some(true),
            },
        };

        let grammar = QwenParser.tool_call_grammar(&[tool], "");
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("\"<parameter=\" \"code\" \">\" \"\\n\"? xml_param_value \"</parameter>\""));
        assert!(!lark.contains("\"\\n</parameter>\""));
    }

    /// LFM3's tokenizer has `<function=` / `<parameter=` (and their closes) as special tokens: the grammar
    /// must name those tokens, or the model writes the tags as text and the special-text guard defuses them.
    #[test]
    fn xml_tags_become_special_tokens_when_the_tokenizer_has_them() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function { name: "read_file".to_string(), description: None, parameters: Some(parameters), strict: None },
        };
        let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
        let eos = u32::try_from(tokens.len()).unwrap();
        tokens.push(b"\xff<eos>".to_vec());
        for t in ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>"] {
            let mut v = b"\xff".to_vec();
            v.extend_from_slice(t.as_bytes());
            tokens.push(v);
        }
        let trie = toktrie::TokTrie::from(&toktrie::TokRxInfo::new(u32::try_from(tokens.len()).unwrap(), eos), &tokens);
        let mut grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        specialize_required_tool_call_grammar(&mut grammar, &trie);
        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains(r#"<[131]> "read_file" ">""#), "<function= by id: {lark}");
        assert!(lark.contains(r#"<[133]> "path" ">""#), "<parameter= by id: {lark}");
        assert!(!lark.contains(r#""<function=""#) && !lark.contains(r#""</parameter>""#), "{lark}");
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        llguidance::ParserFactory::new_simple(&env).unwrap().create_parser(grammar).expect("specialized grammar builds");
    }

    /// A call forced inside a think block first lets the model finish the thought and close it, whether
    /// `</think>` is a special token (HF tokenizer) or plain text (some GGUF vocabularies).
    #[test]
    fn forced_call_inside_reasoning_waits_for_the_close_tag() {
        for special_close in [true, false] {
            let mut tokens = (0_u8..=127).map(|byte| vec![byte]).collect::<Vec<_>>();
            let eos = u32::try_from(tokens.len()).unwrap();
            tokens.push(b"\xff<eos>".to_vec());
            let mut specials = vec!["<tool_call>", "</tool_call>"];
            if special_close {
                specials.push("</think>");
            }
            for t in specials {
                let mut v = b"\xff".to_vec();
                v.extend_from_slice(t.as_bytes());
                tokens.push(v);
            }
            let trie = toktrie::TokTrie::from(&toktrie::TokRxInfo::new(u32::try_from(tokens.len()).unwrap(), eos), &tokens);
            let parameters: HashMap<String, Value> = serde_json::from_value(json!({
                "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"]
            }))
            .unwrap();
            let tool = crate::Tool {
                tp: ToolType::Function,
                function: Function { name: "read_file".to_string(), description: None, parameters: Some(parameters), strict: None },
            };
            let mut grammar = QwenParser.required_tool_call_grammar(&[tool]);
            specialize_required_tool_call_grammar(&mut grammar, &trie);
            assert!(crate::tools::parsers::close_reasoning_before_tool_call(&mut grammar, "</think>", &trie));
            let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
            let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
            let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
            let open_call = trie.get_special_token("<tool_call>").unwrap();
            let mut feed = |text: &[u8]| {
                for token in trie.greedy_tokenize(text) {
                    assert!(matcher.compute_mask().unwrap().is_allowed(token), "special={special_close} rejected {:?}", String::from_utf8_lossy(text));
                    matcher.consume_token(token).unwrap();
                }
            };
            feed(b"user wants <b>the</b> file");
            let mask = matcher.compute_mask().unwrap();
            assert!(!mask.is_allowed(open_call), "special={special_close}: call allowed before the thought closed");
            match trie.get_special_token("</think>") {
                Some(close) => matcher.consume_token(close).map(|_| ()).unwrap(),
                None => {
                    for token in trie.greedy_tokenize(b"</think>") {
                        matcher.consume_token(token).unwrap();
                    }
                }
            }
            assert!(matcher.compute_mask().unwrap().is_allowed(open_call), "special={special_close}: no call after close");
        }
    }

    #[test]
    fn required_grammar_uses_tokenizer_special_wrappers() {
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let mut grammar = QwenParser.required_tool_call_grammar(&[tool]);
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar.clone()).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        assert!(!matcher.compute_mask().unwrap().is_allowed(start_token));

        specialize_required_tool_call_grammar(&mut grammar, factory.tok_env().tok_trie());

        let lark = grammar.grammars[0].lark_grammar.as_ref().unwrap();
        assert!(lark.contains("start: <tool_call> (json_call | xml_call | glm_call)"));
        assert!(lark.contains("json_call: @json_body </tool_call>"));

        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));
        let mask = matcher.compute_mask().unwrap();

        assert!(mask.is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }

    /// `</parameter` inside a value can only become the close: `</parameter=path>` is rejected, while
    /// `</b>` and `</param` stay ordinary value text.
    #[test]
    fn a_value_cannot_hold_a_botched_parameter_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "content": { "type": "string" } },
            "required": ["path"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function { name: "write_file".to_string(), description: None, parameters: Some(parameters), strict: None },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
        let feed = |matcher: &mut llguidance::Matcher, text: &str| {
            for token in trie.greedy_tokenize(text.as_bytes()) {
                assert!(matcher.compute_mask().unwrap().is_allowed(token), "rejected {text:?}");
                matcher.consume_token(token).unwrap();
            }
        };
        feed(&mut matcher, "\n<function=write_file>\n<parameter=content>\n# Report </b> </param x\n</parameter");
        let eq = trie.greedy_tokenize(b"=")[0];
        let gt = trie.greedy_tokenize(b">")[0];
        let mask = matcher.compute_mask().unwrap();
        assert!(!mask.is_allowed(eq), "`</parameter=` must not continue a value");
        assert!(mask.is_allowed(gt));
        feed(&mut matcher, ">\n<parameter=path>\n/tmp/r.md\n</parameter>\n</function>\n");
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.get_special_token("</tool_call>").unwrap()));
    }

    /// LFM3's call, exactly as it emits it (no newline before </tool_call>).
    #[test]
    fn parses_lfm3_xml_call() {
        let out = parse_qwen_tool_calls(
            "<tool_call>\n<function=read_file>\n<parameter=path>/etc/hostname</parameter>\n</function></tool_call>",
        )
        .unwrap()
        .expect("parsed");
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!((v[0]["name"].as_str(), v[0]["arguments"]["path"].as_str()), (Some("read_file"), Some("/etc/hostname")));
    }

    /// Spark's own format (GLM style) parses, values verbatim, `<` inside a value kept.
    #[test]
    fn parses_glm_style_tool_calls() {
        let out = parse_qwen_tool_calls(
            "<tool_call>read_file<arg_key>path</arg_key><arg_value>/etc/hosts</arg_value>\
             <arg_key>note</arg_key><arg_value>a <b> tag</arg_value></tool_call>\
             <tool_call>list_dir<arg_key>limit</arg_key><arg_value>5</arg_value></tool_call>\
             <tool_call>helper_list</tool_call>",
        )
        .unwrap()
        .unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v[0]["name"], "read_file");
        assert_eq!(v[0]["arguments"], json!({ "path": "/etc/hosts", "note": "a <b> tag" }));
        assert_eq!(v[1]["arguments"]["limit"], "5");
        assert_eq!((v[2]["name"].as_str(), &v[2]["arguments"]), (Some("helper_list"), &json!({})));
        // malformed: a key without its value
        assert!(parse_qwen_tool_calls("<tool_call>read_file<arg_key>path</arg_key></tool_call>").unwrap().is_none());
    }

    /// The grammar lets Spark write its native call, and a started `</arg_value` close can only finish.
    #[test]
    fn grammar_accepts_the_glm_form_and_rejects_a_botched_close() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "path": { "type": "string" }, "limit": { "type": "integer" } },
            "required": ["path"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function { name: "read_file".to_string(), description: None, parameters: Some(parameters), strict: None },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(factory.create_parser(grammar).unwrap()));
        let feed = |matcher: &mut llguidance::Matcher, text: &str| {
            for token in trie.greedy_tokenize(text.as_bytes()) {
                assert!(matcher.compute_mask().unwrap().is_allowed(token), "rejected {text:?}");
                matcher.consume_token(token).unwrap();
            }
        };
        feed(&mut matcher, "read_file<arg_key>path</arg_key><arg_value>/etc/<x>hosts</arg_value");
        let eq = trie.greedy_tokenize(b"=")[0];
        let mask = matcher.compute_mask().unwrap();
        assert!(!mask.is_allowed(eq), "`</arg_value=` must not continue a value");
        feed(&mut matcher, "><arg_key>limit</arg_key><arg_value>5</arg_value>");
        assert!(matcher.compute_mask().unwrap().is_allowed(trie.get_special_token("</tool_call>").unwrap()));
    }

    #[test]
    fn continuation_grammar_allows_eos_or_another_call_after_a_block() {
        let parameters: HashMap<String, Value> = serde_json::from_value(json!({
            "type": "object",
            "properties": { "city": { "type": "string" }, "days": { "type": "integer" } },
            "required": ["city"]
        }))
        .unwrap();
        let tool = crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: Some(parameters),
                strict: None,
            },
        };
        let grammar = QwenParser.tool_call_grammar(&[tool], "<tool_call>");
        let trie = wrapper_token_trie();
        let start_token = trie.get_special_token("<tool_call>").unwrap();
        let end_token = trie.get_special_token("</tool_call>").unwrap();
        let eos = trie.eos_token();
        let env: toktrie::TokEnv = Arc::new(GreedyTokenizerEnv { trie: trie.clone() });
        let factory = llguidance::ParserFactory::new_simple(&env).unwrap();
        let parser = factory.create_parser(grammar).unwrap();
        let mut matcher = llguidance::Matcher::new(Ok(parser));

        // Parameters in non-schema order, a `</` inside a value
        let body = "\n<function=get_weather>\n<parameter=days>\n3\n</parameter>\n<parameter=city>\nParis </b>\n</parameter>\n</function>\n";
        for token in trie.greedy_tokenize(body.as_bytes()) {
            assert!(
                matcher.compute_mask().unwrap().is_allowed(token),
                "rejected byte {token}"
            );
            matcher.consume_token(token).unwrap();
        }
        assert!(matcher.compute_mask().unwrap().is_allowed(end_token));
        matcher.consume_token(end_token).unwrap();

        let mask = matcher.compute_mask().unwrap();
        assert!(
            mask.is_allowed(eos),
            "EOS must close the turn after a complete call"
        );
        let newline = trie.greedy_tokenize(b"\n")[0];
        assert!(mask.is_allowed(newline));
        matcher.consume_token(newline).unwrap();
        assert!(matcher.compute_mask().unwrap().is_allowed(start_token));
        matcher.consume_token(start_token).unwrap();
    }
}
