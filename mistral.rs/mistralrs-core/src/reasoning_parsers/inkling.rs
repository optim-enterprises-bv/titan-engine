//! Inkling 1/2 assistant turns: a list of blocks, each closed by `<|end_message|>`:
//! `<|message_model|><|content_thinking|>...` (reasoning), `<|message_model|><|content_text|>...` (content) and
//! `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}` (a tool call, passed to the content
//! verbatim for the Inkling tool-call parser).

pub const HEADER: &str = "<|message_model|>";
pub const THINK: &str = "<|content_thinking|>";
pub const TEXT: &str = "<|content_text|>";
pub const TOOL_JSON: &str = "<|content_invoke_tool_json|>";
pub const END: &str = "<|end_message|>";
/// Longest tool name read before the bytes after a header are taken as plain content.
const MAX_TOOL_NAME: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    /// Between blocks: expecting a header.
    Between,
    /// After a header: expecting a content marker or a tool name.
    Header,
    Think,
    Text,
    Tool,
}

pub struct InklingReasoningContext {
    block: Block,
    buffer: String,
    utf8: Vec<u8>,
    content: String,
    reasoning: String,
    sent_content: usize,
    sent_reasoning: usize,
}

impl Default for InklingReasoningContext {
    fn default() -> Self {
        Self::new()
    }
}

impl InklingReasoningContext {
    pub fn new() -> Self {
        Self {
            block: Block::Between,
            buffer: String::new(),
            utf8: Vec::new(),
            content: String::new(),
            reasoning: String::new(),
            sent_content: 0,
            sent_reasoning: 0,
        }
    }

    /// Byte length of the longest suffix of `s` that is a proper prefix of `tag` (it may complete later).
    fn partial_suffix(s: &str, tag: &str) -> usize {
        (1..tag.len().min(s.len() + 1))
            .rev()
            .find(|&n| s.is_char_boundary(s.len() - n) && tag.starts_with(&s[s.len() - n..]))
            .unwrap_or(0)
    }

    fn sink(&mut self, block: Block) -> &mut String {
        if block == Block::Think {
            &mut self.reasoning
        } else {
            &mut self.content
        }
    }

    pub fn process_text(&mut self, text: &str) {
        self.buffer.push_str(text);
        loop {
            match self.block {
                Block::Between => {
                    let b = self.buffer.trim_start_matches(['\n', ' ']);
                    if b.starts_with(HEADER) {
                        let skip = self.buffer.len() - b.len() + HEADER.len();
                        self.buffer.drain(..skip);
                        self.block = Block::Header;
                    } else if b.is_empty() || HEADER.starts_with(b) {
                        return;
                    } else {
                        // the generation prompt may already hold the first header
                        self.block = Block::Header;
                    }
                }
                Block::Header => {
                    if self.buffer.starts_with(THINK) {
                        self.buffer.drain(..THINK.len());
                        self.block = Block::Think;
                    } else if self.buffer.starts_with(TEXT) {
                        self.buffer.drain(..TEXT.len());
                        self.block = Block::Text;
                    } else if let Some(pos) = self.buffer.find(TOOL_JSON) {
                        let head: String = self.buffer.drain(..pos + TOOL_JSON.len()).collect();
                        self.content.push_str(HEADER);
                        self.content.push_str(&head);
                        self.block = Block::Tool;
                    } else if self.buffer.is_empty()
                        || THINK.starts_with(self.buffer.as_str())
                        || TEXT.starts_with(self.buffer.as_str())
                        || (self.buffer.len() < MAX_TOOL_NAME && !self.buffer.contains(END))
                    {
                        return;
                    } else {
                        // no marker after the header: plain content
                        self.block = Block::Text;
                    }
                }
                block @ (Block::Think | Block::Text | Block::Tool) => {
                    if let Some(pos) = self.buffer.find(END) {
                        let body: String = self.buffer.drain(..pos + END.len()).collect();
                        let body = &body[..pos];
                        self.sink(block).push_str(body);
                        if block == Block::Tool {
                            self.content.push_str(END);
                        }
                        self.block = Block::Between;
                    } else {
                        let keep = Self::partial_suffix(&self.buffer, END);
                        let take = self.buffer.len() - keep;
                        let body: String = self.buffer.drain(..take).collect();
                        self.sink(block).push_str(&body);
                        return;
                    }
                }
            }
        }
    }

    pub fn process_bytes(&mut self, bytes: &[u8]) {
        self.utf8.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.utf8) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid == 0 {
            return;
        }
        let text = String::from_utf8_lossy(&self.utf8[..valid]).into_owned();
        self.utf8.drain(..valid);
        self.process_text(&text);
    }

    pub fn finalize(&mut self) {
        if !self.utf8.is_empty() {
            let text = String::from_utf8_lossy(&self.utf8).into_owned();
            self.utf8.clear();
            self.process_text(&text);
        }
        let rest = std::mem::take(&mut self.buffer);
        match self.block {
            Block::Think => self.reasoning.push_str(&rest),
            Block::Tool => {
                // the stop token may have eaten the close
                self.content.push_str(&rest);
                self.content.push_str(END);
            }
            Block::Between if HEADER.starts_with(rest.trim()) => {}
            _ => self.content.push_str(&rest),
        }
        self.block = Block::Between;
    }

    fn delta(all: &str, sent: &mut usize) -> Option<String> {
        (all.len() > *sent).then(|| {
            let d = all[*sent..].to_string();
            *sent = all.len();
            d
        })
    }
}

impl super::ReasoningParser for InklingReasoningContext {
    fn process_bytes(&mut self, bytes: &[u8]) {
        Self::process_bytes(self, bytes);
    }

    fn finalize(&mut self) {
        Self::finalize(self);
    }

    fn get_content_delta(&mut self) -> Option<String> {
        Self::delta(&self.content, &mut self.sent_content)
    }

    fn get_reasoning_delta(&mut self) -> Option<String> {
        Self::delta(&self.reasoning, &mut self.sent_reasoning)
    }

    fn content(&self) -> Option<String> {
        (!self.content.is_empty()).then(|| self.content.clone())
    }

    fn reasoning_content(&self) -> Option<String> {
        (!self.reasoning.is_empty()).then(|| self.reasoning.clone())
    }

    fn open_block_close_tag(&self) -> Option<&str> {
        (self.block == Block::Think).then_some(END)
    }
}

/// llama.cpp's detection: the Inkling template has both of these.
pub fn is_inkling_template(template: &str) -> bool {
    template.contains("<|message_user|>") && template.contains("<|content_image|>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reasoning_parsers::ReasoningParser;

    fn run(chunks: &[&str]) -> (Option<String>, Option<String>) {
        let mut p = InklingReasoningContext::new();
        for c in chunks {
            p.process_bytes(c.as_bytes());
        }
        ReasoningParser::finalize(&mut p);
        (p.content(), p.reasoning_content())
    }

    #[test]
    fn splits_thinking_text_and_tool_blocks() {
        let out = "<|message_model|><|content_thinking|>Let me look.<|end_message|><|message_model|>read_file<|content_invoke_tool_json|>{\"name\": \"read_file\", \"args\": {\"path\": \"a.py\"}}<|end_message|>";
        let chunks: Vec<String> = out.chars().collect::<Vec<_>>().chunks(3).map(|c| c.iter().collect()).collect();
        let refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
        let (content, reasoning) = run(&refs);
        assert_eq!(reasoning.as_deref(), Some("Let me look."));
        assert_eq!(
            content.as_deref(),
            Some("<|message_model|>read_file<|content_invoke_tool_json|>{\"name\": \"read_file\", \"args\": {\"path\": \"a.py\"}}<|end_message|>")
        );
    }

    #[test]
    fn text_block_loses_markers() {
        let (content, reasoning) = run(&["<|message_model|><|content_text|>Hello", " there<|end_", "message|>"]);
        assert_eq!(content.as_deref(), Some("Hello there"));
        assert_eq!(reasoning, None);
    }

    #[test]
    fn unclosed_tool_block_is_closed_at_finalize() {
        let (content, _) = run(&["<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}"]);
        assert_eq!(content.as_deref(), Some("<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{}}<|end_message|>"));
    }

    #[test]
    fn missing_first_header() {
        let (content, reasoning) = run(&["<|content_thinking|>hmm<|end_message|><|message_model|><|content_text|>ok<|end_message|>"]);
        assert_eq!(reasoning.as_deref(), Some("hmm"));
        assert_eq!(content.as_deref(), Some("ok"));
    }
}
