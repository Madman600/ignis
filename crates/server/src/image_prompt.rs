//! The image model's prompt (spec image-model/03 §Tokenizer and template):
//! the pipeline's raw template, tokenized with the artifact's tokenizer.
//!
//! diffusers' `QwenImage21Pipeline` does not render its prompt through the
//! chat template: it formats a raw template string and tokenizes that, and
//! the two differ once there is a condition image (`<image1>` before
//! `<|vision_start|>`). Everything here comes from the artifact: its
//! tokenizer, its chat template, and `image_model.json`'s `prompt` (verbatim
//! from diffusers 0.41.0):
//! - text-to-image formats `template_t2i`; with N condition images,
//!   `template_ti2i`'s first image slot becomes N slots (`image_slot`
//!   numbered from 1, joined by `image_slot_separator`);
//! - an empty prompt becomes `empty_prompt` (" "): Qwen has no BOS, so the
//!   encoder would read nothing;
//! - each slot's `<|image_pad|>` becomes as many pads as its image has merged
//!   tokens, as the processor expands it;
//! - `drop_idx`, the leading rows the encoder's output loses, is the system
//!   message's length as the chat template tokenizes it, computed at load as
//!   the pipeline computes it.

use serde_json::Value as JsonValue;

use ignis_artifact::vision::IMAGE_PAD;
use ignis_artifact::{ChatMessage, ChatTemplate, ContentPart, MessageContent, Role, Tokenizer};

/// The artifact's tokenizer and prompt templates, with `drop_idx`.
pub struct ImagePrompt {
    tokenizer: Tokenizer,
    template_t2i: String,
    template_ti2i: String,
    image_slot: String,
    image_slot_separator: String,
    empty_prompt: String,
    drop_idx: usize,
    image_pad: u32,
}

impl ImagePrompt {
    /// From the artifact's tokenizer, chat template and `image_model.json`'s
    /// `prompt`.
    pub fn new(tokenizer: Tokenizer, chat_template: &ChatTemplate, prompt: &JsonValue) -> Result<Self, String> {
        let field = |name: &str| {
            prompt.get(name).and_then(JsonValue::as_str).map(str::to_owned).ok_or_else(|| format!("image_model.json: prompt.{name} is missing"))
        };
        // The pipeline tokenizes [{"role": "system", "content": [{"type": "text", "text": system}]}]
        // through the chat template without a generation prompt; `render` always adds one.
        let system = ChatMessage {
            role: Role::System,
            content: MessageContent::Parts(vec![ContentPart::Text(field("system")?)]),
            tool_calls: Vec::new(),
            reasoning_content: None,
        };
        let rendered = chat_template.render(&[system]).map_err(|e| e.to_string())?;
        let system = rendered
            .strip_suffix(ChatTemplate::GENERATION_OPENER)
            .ok_or_else(|| format!("the chat template's system message does not end with a generation prompt: {rendered:?}"))?;
        let drop_idx = tokenizer.encode(system).map_err(|e| e.to_string())?.len();
        let image_pad = match tokenizer.encode(IMAGE_PAD).map_err(|e| e.to_string())?[..] {
            [id] => id,
            ref ids => return Err(format!("the tokenizer splits {IMAGE_PAD} into {ids:?}")),
        };
        let out = Self {
            tokenizer,
            template_t2i: field("template_t2i")?,
            template_ti2i: field("template_ti2i")?,
            image_slot: field("image_slot")?,
            image_slot_separator: field("image_slot_separator")?,
            empty_prompt: field("empty_prompt")?,
            drop_idx,
            image_pad,
        };
        // drop_idx counts the templates' own first rows, and the slot replacement must find its slot
        for template in [&out.template_t2i, &out.template_ti2i] {
            if !template.starts_with(system) || !template.contains("{}") {
                return Err(format!("image_model.json: the template {template:?} does not open with the system message and hold the prompt's {{}}"));
            }
        }
        if !out.template_ti2i.contains(&out.slot(1)) {
            return Err(format!("image_model.json: template_ti2i has no {:?}", out.slot(1)));
        }
        Ok(out)
    }

    /// The rows the encoder's output loses: the system message.
    pub fn drop_idx(&self) -> usize {
        self.drop_idx
    }

    /// The `<|image_pad|>` id.
    pub fn image_pad(&self) -> u32 {
        self.image_pad
    }

    /// Condition image `image`'s slot (from 1).
    fn slot(&self, image: usize) -> String {
        self.image_slot.replacen("{}", &image.to_string(), 1)
    }

    /// The raw template's text for `prompt` with `images` condition images,
    /// one slot each: the pipeline's string, before the processor expands the
    /// pads.
    pub fn text(&self, prompt: &str, images: usize) -> String {
        let prompt = if prompt.is_empty() { &self.empty_prompt } else { prompt };
        let template = if images == 0 {
            self.template_t2i.clone()
        } else {
            let slots: Vec<String> = (1..=images).map(|i| self.slot(i)).collect();
            self.template_ti2i.replace(&self.slot(1), &slots.join(&self.image_slot_separator))
        };
        template.replacen("{}", prompt, 1)
    }

    /// The rows of `prompt` with `images` condition images before the pads are
    /// expanded: the raw template tokenized, one pad a slot.
    pub fn rows(&self, prompt: &str, images: usize) -> Result<usize, String> {
        Ok(self.tokenizer.encode(&self.text(prompt, images)).map_err(|e| e.to_string())?.len())
    }

    /// The token ids of `prompt` with condition images of `image_tokens`
    /// merged tokens each: the raw template tokenized, each slot's pad
    /// repeated for its image. A prompt that writes an image pad itself is
    /// refused: the pad would take an image's place.
    pub fn tokens(&self, prompt: &str, image_tokens: &[usize]) -> Result<Vec<u32>, String> {
        let ids = self.tokenizer.encode(&self.text(prompt, image_tokens.len())).map_err(|e| e.to_string())?;
        if ids.iter().filter(|&&id| id == self.image_pad).count() != image_tokens.len() {
            return Err(format!("the prompt contains {IMAGE_PAD}"));
        }
        let mut counts = image_tokens.iter();
        let mut out = Vec::with_capacity(ids.len() + image_tokens.iter().sum::<usize>());
        for id in ids {
            if id == self.image_pad {
                out.extend(std::iter::repeat(id).take(*counts.next().unwrap()));
            } else {
                out.push(id);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A word-level tokenizer with the template's special tokens split out
    /// first, as the real one does; every other word is `x`.
    fn tokenizer() -> Tokenizer {
        let special = ["<|im_start|>", "<|im_end|>", IMAGE_PAD, "<|vision_start|>", "<|vision_end|>"];
        let added: Vec<JsonValue> = (1..).zip(special).map(|(id, content)| serde_json::json!({
            "id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
            "normalized": false, "special": true,
        })).collect();
        let mut vocab = serde_json::json!({ "x": 0 });
        for (id, content) in (1..).zip(special) {
            vocab[content] = id.into();
        }
        let json = serde_json::json!({
            "version": "1.0", "added_tokens": added, "pre_tokenizer": { "type": "Whitespace" },
            "model": { "type": "WordLevel", "vocab": vocab, "unk_token": "x" },
        });
        Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap()
    }

    /// The real chat template's system branch, and its generation prompt.
    const CHAT_TEMPLATE: &str = r"{%- for message in messages %}{{- '<|im_start|>' + message.role + '\n' }}
        {%- for content in message.content %}{{- content.text }}{%- endfor %}{{- '<|im_end|>\n' }}{%- endfor %}
        {%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}{%- endif %}";

    /// The checkpoint's templates over a word-level tokenizer.
    pub(crate) fn prompt() -> ImagePrompt {
        let system = "Comprehend and analyze the provided prompt.";
        let json = serde_json::json!({
            "system": system,
            "template_t2i": format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"),
            "template_ti2i": format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{{}}<|im_end|>\n<|im_start|>assistant\n"),
            "image_slot": "<image{}><|vision_start|><|image_pad|><|vision_end|>",
            "image_slot_separator": " ",
            "empty_prompt": " ",
        });
        ImagePrompt::new(tokenizer(), &ChatTemplate::from_source(CHAT_TEMPLATE).unwrap(), &json).unwrap()
    }

    #[test]
    fn drop_idx_is_the_system_message_without_a_generation_prompt() {
        // <|im_start|> system Comprehend and analyze the provided prompt . <|im_end|>
        assert_eq!(prompt().drop_idx(), 10);
    }

    #[test]
    fn condition_images_get_one_numbered_slot_each() {
        let p = prompt();
        let slot = |i: usize| format!("<image{i}><|vision_start|><|image_pad|><|vision_end|>");
        assert!(p.text("fox", 0).ends_with("<|im_start|>user\nfox<|im_end|>\n<|im_start|>assistant\n"));
        assert!(p.text("fox", 1).contains(&format!("user\n{}fox<|im_end|>", slot(1))));
        assert!(p.text("fox", 2).contains(&format!("user\n{} {}fox<|im_end|>", slot(1), slot(2))));
        let ten: Vec<String> = (1..=10).map(slot).collect();
        assert!(p.text("fox", 10).contains(&format!("user\n{}fox<|im_end|>", ten.join(" "))));
        // the prompt is inserted verbatim, braces and slots included
        assert!(p.text("{} <image1>", 1).contains(&format!("{}{{}} <image1><|im_end|>", slot(1))));
    }

    #[test]
    fn an_empty_prompt_becomes_a_space() {
        let p = prompt();
        assert_eq!(p.text("", 0), p.text(" ", 0));
        assert_eq!(p.tokens("", &[]).unwrap(), p.tokens(" ", &[]).unwrap());
    }

    #[test]
    fn each_slot_pad_becomes_its_images_merged_tokens() {
        let p = prompt();
        let ids = p.tokens("fox", &[3, 2]).unwrap();
        let pad = p.image_pad();
        let runs: Vec<usize> = ids.split(|&id| id != pad).map(<[u32]>::len).filter(|&n| n > 0).collect();
        assert_eq!(runs, [3, 2]);
        assert_eq!(ids.len(), p.tokens("fox", &[1, 1]).unwrap().len() + 3);
        assert!(p.tokens("a <|image_pad|>", &[]).is_err(), "a pad in the prompt");
        assert!(p.tokens("a <|image_pad|>", &[4]).is_err(), "a pad in the prompt, with an image");
    }
}
