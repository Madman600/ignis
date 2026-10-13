"""The prompt templates the artifact carries are diffusers' own, verbatim."""
import inspect
import textwrap
from types import SimpleNamespace

import metadata


def _fragment(source, start, end):
    """The lines of `source` from the one holding `start` up to the one holding `end`."""
    first = source.rfind("\n", 0, source.index(start)) + 1
    return textwrap.dedent(source[first:source.rfind("\n", 0, source.index(end)) + 1])


def test_the_templates_are_the_pipelines():
    from diffusers import QwenImage21Pipeline
    init = inspect.getsource(QwenImage21Pipeline.__init__)
    pipe = SimpleNamespace()
    exec(_fragment(init, "self.sys_prompt =", "# Number of leading system-role tokens"), {"self": pipe})
    assert pipe.sys_prompt == metadata.SYSTEM_PROMPT
    assert pipe.prompt_template_t2i == metadata.TEMPLATE_T2I
    assert pipe.prompt_template_ti2i == metadata.TEMPLATE_TI2I
    embeds = inspect.getsource(QwenImage21Pipeline._get_qwen_prompt_embeds)
    # condition image i's slot, joined by one space; an empty prompt becomes " "
    assert 'replace = "' + metadata.IMAGE_SLOT.format(1) + '"' in embeds
    assert ('replace += f"' + metadata.IMAGE_SLOT_SEPARATOR + metadata.IMAGE_SLOT.replace("{}", "{i}") + '"') in embeds
    assert f'prompt = ["{metadata.EMPTY_PROMPT}" if not p else p for p in prompt]' in embeds
