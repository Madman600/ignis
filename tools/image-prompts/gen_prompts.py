"""Writes crates/server/src/images/prompts.rs from the pinned QwenLM/Qwen-Image
sources (argv[1] = the directory holding prompt_utils_2512.py and
prompt_utils.py, argv[2] = the output file). Each prompt is the string
literal's runtime value (escapes resolved), trimmed of the surrounding
whitespace and nothing else.

Recipe (spec image-model/05 §Prompt enhancement), from the repository root:

    mkdir -p .scratch/upstream/qwen-image
    base=https://raw.githubusercontent.com/QwenLM/Qwen-Image/6b5e1f5cec987d404be5ac6657db3b9aacb56a89
    for f in prompt_utils_2512.py prompt_utils.py; do
      curl -fsSL -o .scratch/upstream/qwen-image/$f $base/src/examples/tools/$f
    done
    python tools/image-prompts/gen_prompts.py .scratch/upstream/qwen-image crates/server/src/images/prompts.rs

At the pinned commit it writes the file byte for byte as committed. A re-pin
changes the commit in the URL and in the header below together."""
import ast
import sys

src, out = sys.argv[1], sys.argv[2]


def assigned(path, function, name):
    tree = ast.parse(open(path, encoding='utf-8').read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(getattr(t, 'id', None) == name for t in stmt.targets):
                    return stmt.value.value
    raise SystemExit(f"{function}.{name} not found in {path}")


def raw(text):
    hashes = 1
    while '"' + '#' * hashes in text:
        hashes += 1
    return 'r' + '#' * hashes + '"' + text + '"' + '#' * hashes


t2i_en = assigned(src + '/prompt_utils_2512.py', 'polish_prompt_en', 'SYSTEM_PROMPT').strip()
t2i_zh = assigned(src + '/prompt_utils_2512.py', 'polish_prompt_zh', 'SYSTEM_PROMPT').strip()
edit = assigned(src + '/prompt_utils.py', 'polish_edit_prompt', 'EDIT_SYSTEM_PROMPT').strip()

header = '''//! The system prompts prompt enhancement rewrites with (spec image-model/05
//! §Prompt enhancement): Qwen's own, from QwenLM/Qwen-Image at commit
//! 6b5e1f5cec987d404be5ac6657db3b9aacb56a89, Copyright 2024 Alibaba Cloud,
//! under the Apache License 2.0 (the root NOTICE says so):
//!
//! - [`GENERATION_EN`] and [`GENERATION_ZH`]: `src/examples/tools/prompt_utils_2512.py`,
//!   `polish_prompt_en`'s and `polish_prompt_zh`'s `SYSTEM_PROMPT`;
//! - [`EDIT`]: `src/examples/tools/prompt_utils.py`, `polish_edit_prompt`'s
//!   `EDIT_SYSTEM_PROMPT`.
//!
//! Adapted: each is the literal's text with its surrounding whitespace
//! trimmed and nothing else changed, and it goes in the system turn — upstream
//! puts it at the head of the user turn, after the generic system prompt of
//! the hosted API it calls. The user turn is upstream's own
//! ([`super::enhance`]). Each constant is generated from the pinned file:
//! the literal's runtime value, its escapes resolved; a change to one is a
//! re-pin of the commit above, this whole file written again by
//! `tools/image-prompts/gen_prompts.py` (its docstring has the recipe).

'''

body = header
body += '/// Rewrites a generation\'s English prompt (`polish_prompt_en`).\n'
body += 'pub const GENERATION_EN: &str = ' + raw(t2i_en) + ';\n\n'
body += '/// Rewrites a generation\'s prompt that holds a CJK ideograph, in Chinese\n/// (`polish_prompt_zh`).\n'
body += 'pub const GENERATION_ZH: &str = ' + raw(t2i_zh) + ';\n\n'
body += '/// Rewrites an edit\'s instruction (`polish_edit_prompt`): its answer is\n/// `{"Rewritten": "..."}`, maybe fenced as JSON.\n'
body += 'pub const EDIT: &str = ' + raw(edit) + ';\n'
open(out, 'wb').write(body.encode('utf-8'))
print(len(t2i_en), len(t2i_zh), len(edit))
