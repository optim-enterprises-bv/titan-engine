use std::sync::Arc;

use anyhow::Result;
use llguidance::{api::TopLevelGrammar, ParserFactory};
use tokenizers::{decoders::DecoderWrapper, Tokenizer};

use crate::Constraint;

pub fn build_llg_factory(mut tokenizer: Tokenizer) -> Result<Arc<ParserFactory>> {
    let decoder = match tokenizer.get_decoder() {
        Some(DecoderWrapper::Sequence(sequence)) if sequence.get_decoders().len() == 1 => {
            match &sequence.get_decoders()[0] {
                DecoderWrapper::ByteLevel(decoder) => Some(DecoderWrapper::ByteLevel(*decoder)),
                _ => None,
            }
        }
        _ => None,
    };
    if let Some(decoder) = decoder {
        tokenizer.with_decoder(Some(decoder));
    }

    // Collect special token info before from_tokenizer() consumes the tokenizer.
    // Tokens the grammars reference as literals, not only those the tokenizer flags `special`.
    // LFM3 ships `<tool_call>`, `</tool_call>`, `<function=`, `</function>`, `<parameter=` as
    // USER_DEFINED added tokens (single ids 125020-125022, `special: false`), so the tool-call
    // grammar could not resolve `</tool_call>` and every constrained decode silently fell back
    // to unconstrained generation ("Failed to build tool call grammar: unknown special token").
    // Marker-shaped added tokens are registered here so grammars can address them by name.
    let added_special: Vec<(u32, String)> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .map(|(id, at)| (id, at))
        .filter(|(_, at)| {
            at.special || (at.content.starts_with('<') && at.content.ends_with('>'))
        })
        .map(|(id, at)| (id, at.content))
        .collect();

    // Tool-call markers must be addressable by the grammars (`<token>"</tool_call>"<`), which
    // requires the SPECIAL_TOKEN_MARKER prefix in the trie. LFM3 keeps `<tool_call>`,
    // `</tool_call>`, `<function=`, `</function>`, `<parameter=`, `</parameter>` in the *base*
    // vocab (USER_DEFINED, not flagged added/special), so the added-token pass above never sees
    // them and every tool-call grammar failed to build. Resolve each literal by encoding it and
    // mark the id directly; a marker that is not a single token stays untouched.
    // Tool-call markers must be resolvable by the grammars (`<token>"</tool_call>"<`), which needs
    // the SPECIAL_TOKEN_MARKER prefix in the trie. LFM3 keeps `<tool_call>`, `</tool_call>`,
    // `<function=`, `</function>`, `<parameter=`, `</parameter>` in the base vocab, unflagged, and
    // the strings carry zero-width characters — the plain ASCII literals never match a vocab key,
    // which is why every tool-call grammar failed to build and titan silently decoded unconstrained.
    // Match on the zero-width-stripped form and keep the key's exact bytes.
    const MARKERS: [&str; 6] = [
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    ];
    let strip_zw = |s: &str| -> String {
        s.chars()
            .filter(|c| !matches!(*c, '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'))
            .collect()
    };
    let mut marker_ids: Vec<(usize, String)> = Vec::new();
    for (tok, id) in tokenizer.get_vocab(true) {
        let stripped = strip_zw(&tok);
        if MARKERS.contains(&stripped.as_str()) {
            marker_ids.push((id as usize, tok.clone()));
        }
    }
    if marker_ids.is_empty() {
        tracing::warn!("no tool-call marker tokens found in the vocab; tool grammars will not build");
    } else {
        tracing::info!(
            "tool-call marker tokens: {:?}",
            marker_ids.iter().map(|(i, t)| (i, strip_zw(t))).collect::<Vec<_>>()
        );
    }

    let bt = toktrie_hf_tokenizers::ByteTokenizer::from_tokenizer(tokenizer)?;
    let info = bt.tokrx_info();
    let mut token_bytes = bt.token_bytes();

    // Fix up special tokens that from_tokenizer() may have missed.
    // Tekken tokenizers add special tokens to the BPE base vocab first, which can
    // prevent the toktrie builder from applying the SPECIAL_TOKEN_MARKER prefix.
    for (id, content) in &added_special {
        let idx = *id as usize;
        if idx < token_bytes.len()
            && (token_bytes[idx].is_empty()
                || token_bytes[idx][0] != toktrie::TokTrie::SPECIAL_TOKEN_MARKER)
        {
            let mut bytes = content.as_bytes().to_vec();
            bytes.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            token_bytes[idx] = bytes;
        }
    }

    for (idx, marker) in &marker_ids {
        if *idx < token_bytes.len() {
            let mut bytes = marker.as_bytes().to_vec();
            bytes.insert(0, toktrie::TokTrie::SPECIAL_TOKEN_MARKER);
            token_bytes[*idx] = bytes;
        }
    }
    tracing::info!("marked {} tool-call marker tokens in the token trie", marker_ids.len());

    let tok_trie = toktrie::TokTrie::from(&info, &token_bytes);
    let env = toktrie_hf_tokenizers::ByteTokenizerEnv {
        tokenizer: bt,
        tok_trie,
    }
    .to_env();
    let factory = ParserFactory::new_simple(&env)?;
    Ok(Arc::new(factory))
}

pub fn llg_grammar_from_constraint(constraint: &Constraint) -> Result<Option<TopLevelGrammar>> {
    let grm = match constraint {
        Constraint::Regex(regex) => TopLevelGrammar::from_regex(regex),
        Constraint::Lark(lark) => TopLevelGrammar::from_lark(lark.clone()),
        Constraint::JsonSchema(value) => TopLevelGrammar::from_json_schema(value.clone()),
        Constraint::Llguidance(value) => value.clone(),
        Constraint::None => return Ok(None),
    };
    Ok(Some(grm))
}

pub fn constraint_from_llg_grammar(
    factory: &ParserFactory,
    grm: TopLevelGrammar,
) -> Result<llguidance::Matcher> {
    let parser = factory.create_parser(grm)?;
    Ok(llguidance::Matcher::new(Ok(parser)))
}
