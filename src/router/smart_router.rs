use crate::types::chat::{ChatCompletionRequest, Message};

/// Task classification result with richer metadata for routing decisions.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TaskClassification {
    /// Primary category: tool_call, code, math, translation, general
    pub category: &'static str,
    /// Whether the request includes tool definitions
    pub has_tools: bool,
    /// Estimated complexity: low, medium, high
    pub complexity: &'static str,
    /// Number of messages in the conversation
    pub message_count: usize,
}

/// Classify a chat completion request into a task type.
pub fn classify_request(req: &ChatCompletionRequest) -> TaskClassification {
    let has_tools = req.tools.as_ref().is_some_and(|t| !t.is_empty());
    let message_count = req.messages.len();

    // Tool call requests get their own category
    if has_tools {
        return TaskClassification {
            category: "tool_call",
            has_tools,
            complexity: estimate_complexity(message_count, has_tools),
            message_count,
        };
    }

    let category = classify_messages(&req.messages);
    TaskClassification {
        category,
        has_tools,
        complexity: estimate_complexity(message_count, has_tools),
        message_count,
    }
}

/// Classify based on message content (original logic, enhanced).
fn classify_messages(messages: &[Message]) -> &'static str {
    let text = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.content.as_ref())
        .map(|c| c.as_text())
        .unwrap_or_default();

    let text_lower = text.to_lowercase();

    // 1. Code detection
    if text.contains("```")
        || has_any_word(
            &text_lower,
            &[
                "function", "import ", "const ", "var ", "fn ", "pub ", "async",
                "=>", "compile", "runtime", "debug",
            ],
        )
        || has_any_word(&text, &["def ", "class "])
        || has_any_word(&text_lower, &["error", "bug", "fix"])
    {
        return "code";
    }

    // 2. Math detection
    if has_any_char(&text, &['∑', '∫', '∂', '√'])
        || has_any_word(&text, &["\\frac", "\\sum", "\\int"])
        || (text.contains('$') && text.matches('$').count() >= 2)
        || has_any_word(
            &text_lower,
            &["equation", "theorem", "proof", "calculate"],
        )
        || has_any_word(&text, &["公式", "方程", "计算"])
    {
        return "math";
    }

    // 3. Translation detection
    if has_any_word(&text_lower, &["translate", "translation"]) || text.contains("翻译") {
        return "translation";
    }

    // 4. Default
    "general"
}

fn estimate_complexity(message_count: usize, has_tools: bool) -> &'static str {
    if has_tools || message_count > 10 {
        "high"
    } else if message_count > 4 {
        "medium"
    } else {
        "low"
    }
}

fn has_any_word(text: &str, words: &[&str]) -> bool {
    words.iter().any(|w| text.contains(w))
}

fn has_any_char(text: &str, chars: &[char]) -> bool {
    chars.iter().any(|c| text.contains(*c))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::chat::{Message, MessageContent};

    fn msg(role: &str, content: &str) -> Message {
        Message {
            role: role.to_string(),
            content: Some(MessageContent::Text(content.to_string())),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn make_req(messages: Vec<Message>, with_tools: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages,
            temperature: None,
            top_p: None,
            max_tokens: None,
            stream: false,
            stop: None,
            tools: if with_tools {
                Some(vec![crate::types::chat::Tool {
                    tool_type: "function".to_string(),
                    function: crate::types::chat::FunctionDef {
                        name: "test_fn".to_string(),
                        description: Some("test".to_string()),
                        parameters: None,
                    },
                }])
            } else {
                None
            },
            tool_choice: None,
        }
    }

    #[test]
    fn test_classify_tool_call() {
        let req = make_req(vec![msg("user", "Hello")], true);
        let c = classify_request(&req);
        assert_eq!(c.category, "tool_call");
        assert!(c.has_tools);
    }

    #[test]
    fn test_classify_code() {
        let req = make_req(
            vec![msg("user", "Help me write a Python function to sort a list")],
            false,
        );
        assert_eq!(classify_request(&req).category, "code");

        let req = make_req(vec![msg("user", "```rust\nfn main() {}\n```")], false);
        assert_eq!(classify_request(&req).category, "code");

        let req = make_req(vec![msg("user", "I have a bug in my code")], false);
        assert_eq!(classify_request(&req).category, "code");
    }

    #[test]
    fn test_classify_math() {
        let req = make_req(
            vec![msg("user", "Solve this equation: x² + 2x + 1 = 0")],
            false,
        );
        assert_eq!(classify_request(&req).category, "math");

        let req = make_req(vec![msg("user", "请帮我解这个方程")], false);
        assert_eq!(classify_request(&req).category, "math");
    }

    #[test]
    fn test_classify_translation() {
        let req = make_req(vec![msg("user", "Translate this to Chinese")], false);
        assert_eq!(classify_request(&req).category, "translation");
    }

    #[test]
    fn test_classify_general() {
        let req = make_req(vec![msg("user", "今天天气怎么样")], false);
        assert_eq!(classify_request(&req).category, "general");
    }

    #[test]
    fn test_complexity_estimation() {
        // Tool calls → high
        let req = make_req(vec![msg("user", "Hi")], true);
        assert_eq!(classify_request(&req).complexity, "high");

        // Many messages → high
        let msgs: Vec<Message> = (0..12).map(|i| msg("user", &format!("msg {i}"))).collect();
        let req = make_req(msgs, false);
        assert_eq!(classify_request(&req).complexity, "high");

        // Few messages → low
        let req = make_req(vec![msg("user", "Hello")], false);
        assert_eq!(classify_request(&req).complexity, "low");
    }
}
