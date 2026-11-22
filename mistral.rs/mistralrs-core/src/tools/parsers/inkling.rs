//! Inkling tool calls: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`,
//! one block per call (the Inkling reasoning parser passes these blocks to the content verbatim).

use llguidance::api::TopLevelGrammar;
use serde_json::{json, Value};

use super::ToolFormatParser;
use crate::reasoning_parsers::inkling::{END, HEADER, TOOL_JSON};
use crate::Tool;

pub struct InklingParser;

impl ToolFormatParser for InklingParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains(TOOL_JSON) || text.contains(HEADER) || HEADER.starts_with(text.trim_start()) && !text.trim().is_empty()
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Inkling
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        self.required_tool_call_grammar(tools)
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(r#"start: @json_body"#.to_string(), tools, "args", true)
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let mut calls = Vec::new();
        let mut rest = message;
        while let Some(pos) = rest.find(TOOL_JSON) {
            let name = rest[..pos].rsplit(HEADER).next().unwrap_or("").trim().to_string();
            let after = &rest[pos + TOOL_JSON.len()..];
            let Some(end) = after.find(END) else {
                return Ok(None);
            };
            let raw = after[..end].trim();
            let Some(body) = serde_json::from_str::<Value>(raw)
                .ok()
                .or_else(|| serde_json::from_str(&close_unbalanced(&escape_controls_in_strings(raw))).ok())
            else {
                tracing::warn!("inkling: unparseable tool call body ({} bytes): {}", raw.len(), raw.chars().take(600).collect::<String>());
                return Ok(None);
            };
            let name = body.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or(name);
            let args = body.get("args").or_else(|| body.get("arguments")).cloned().unwrap_or_else(|| json!({}));
            calls.push(json!({ "name": name, "arguments": args }));
            rest = &after[end + END.len()..];
        }
        if calls.is_empty() {
            return Ok(None);
        }
        Ok(Some(Value::Array(calls).to_string()))
    }
}

/// Unconstrained generation can put raw newlines and tabs inside JSON strings (code payloads); escape them.
fn escape_controls_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let (mut in_str, mut esc) = (false, false);
    for c in s.chars() {
        if in_str && !esc && (c as u32) < 0x20 {
            match c {
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c => out.push_str(&format!("\\u{:04x}", c as u32)),
            }
            continue;
        }
        out.push(c);
        if esc {
            esc = false;
        } else if in_str && c == '\\' {
            esc = true;
        } else if c == '"' {
            in_str = !in_str;
        }
    }
    out
}

/// Unconstrained generation sometimes drops the wrapper's last brace (llama.cpp's grammar emits it); append the
/// closers of the objects and arrays left open, outside strings.
fn close_unbalanced(s: &str) -> String {
    let mut open = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    for c in s.chars() {
        if esc {
            esc = false;
        } else if in_str {
            match c {
                '\\' => esc = true,
                '"' => in_str = false,
                _ => {}
            }
        } else {
            match c {
                '"' => in_str = true,
                '{' => open.push('}'),
                '[' => open.push(']'),
                '}' | ']' => {
                    open.pop();
                }
                _ => {}
            }
        }
    }
    let mut out = s.to_string();
    out.extend(open.into_iter().rev());
    out
}

#[cfg(test)]
mod tests {
    use super::InklingParser;
    use crate::tools::parsers::{extract_model_specific_message, ToolFormatParser};

    #[test]
    fn parses_calls_and_keeps_text() {
        let m = "Checking.<|message_model|>read_file<|content_invoke_tool_json|>{\"name\": \"read_file\", \"args\": {\"path\": \"a.py\"}}<|end_message|><|message_model|>ls<|content_invoke_tool_json|>{\"name\":\"ls\",\"args\":{}}<|end_message|>";
        let parsed: serde_json::Value = serde_json::from_str(&InklingParser.parse(m).unwrap().unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!([{"name": "read_file", "arguments": {"path": "a.py"}}, {"name": "ls", "arguments": {}}]));
        let (_, content) = extract_model_specific_message(m).unwrap().unwrap();
        assert_eq!(content, "Checking.");
    }

    #[test]
    fn raw_newlines_in_strings_are_repaired() {
        let m = "<|message_model|>w<|content_invoke_tool_json|>{\"name\":\"w\",\"args\":{\"code\":\"a\n\tb \\\"q\\\"\"}}<|end_message|>";
        let parsed: serde_json::Value = serde_json::from_str(&InklingParser.parse(m).unwrap().unwrap()).unwrap();
        assert_eq!(parsed[0]["arguments"]["code"], "a\n\tb \"q\"");
    }

    #[test]
    fn missing_wrapper_brace_is_repaired() {
        let m = "<|message_model|>w<|content_invoke_tool_json|>{\"name\":\"w\",\"args\":{\"p\":{\"files\":[{\"x\":\"}\"}]}}<|end_message|>";
        let parsed: serde_json::Value = serde_json::from_str(&InklingParser.parse(m).unwrap().unwrap()).unwrap();
        assert_eq!(parsed[0]["arguments"]["p"]["files"][0]["x"], "}");
    }

    #[test]
    fn incomplete_call_is_not_parsed() {
        assert!(InklingParser.parse("<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\"").unwrap().is_none());
    }
}
