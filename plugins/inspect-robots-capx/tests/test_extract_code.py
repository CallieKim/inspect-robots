"""Code extraction from model replies, including the tool-call markup real models emit."""

from __future__ import annotations

import pytest

from inspect_robots_agent import AssistantMessage
from inspect_robots_capx.policy import _control_word, _extract_code

CODE = 'import numpy as np\nr = segment("red cube")\nmove_to_joints(solve_ik(g, q))\n'


def _extract(reply: str) -> str:
    return _extract_code(AssistantMessage(content=reply, tool_calls=()))


def test_bare_code_is_used_as_is() -> None:
    assert _extract(CODE) == CODE.strip()


def test_fenced_code_is_unwrapped() -> None:
    assert _extract(f"```python\n{CODE}```") == CODE.rstrip("\n")


def test_first_fenced_block_wins_over_surrounding_prose() -> None:
    assert _extract(f"Here is the plan.\n```python\n{CODE}```\nDone.") == CODE.rstrip("\n")


def test_regenerate_prefix_is_dropped() -> None:
    assert _extract(f"REGENERATE\n{CODE}") == CODE.strip()


def test_single_tool_call_block_yields_its_code() -> None:
    reply = f'<invoke name="python">\n<parameter name="code">{CODE}</parameter>\n</invoke>\n'
    assert _extract(reply) == CODE.strip("\n")


@pytest.mark.parametrize(
    ("tool", "parameter"), [("bash", "command"), ("python", "code"), ("None", "code")]
)
def test_tool_and_parameter_names_do_not_matter(tool: str, parameter: str) -> None:
    reply = f'<invoke name="{tool}">\n<parameter name="{parameter}">{CODE}</parameter>\n</invoke>'
    assert _extract(reply) == CODE.strip("\n")


def test_invented_continuation_after_the_first_block_is_discarded() -> None:
    reply = (
        f'<invoke name="python">\n<parameter name="code">{CODE}</parameter>\n</invoke>\n'
        "actually\n\n"
        '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</invoke>\n'
        "Executed. stdout:\nc=[ 0.0572 -0.2735  0.0465]\n"
        "FINISH"
    )
    assert _extract(reply) == CODE.strip("\n")


def test_unterminated_block_runs_to_the_end_of_the_reply() -> None:
    assert _extract('<invoke name="python">\n<parameter name="code">print(1)') == "print(1)"


def test_regenerate_before_a_tool_call_block() -> None:
    reply = (
        f'REGENERATE\n<invoke name="python">\n<parameter name="code">{CODE}</parameter>\n</invoke>'
    )
    assert _extract(reply) == CODE.strip("\n")


def test_prose_before_a_tool_call_block_is_ignored() -> None:
    block = f'<invoke name="python">\n<parameter name="code">{CODE}</parameter>\n</invoke>'
    reply = f"I will pick it up.\n{block}"
    assert _extract(reply) == CODE.strip("\n")


def test_comparison_operators_inside_the_code_survive() -> None:
    body = "if c[0] < 0.1 and c[1] > -0.3:\n    open_gripper()\n"
    reply = f'<invoke name="python">\n<parameter name="code">{body}</parameter>\n</invoke>'
    assert _extract(reply) == body.strip("\n")


def test_markup_free_reply_mentioning_invoke_is_left_alone() -> None:
    reply = "invoke = 3\nprint(invoke)"
    assert _extract(reply) == reply


_BLOCK = '<invoke name="python">\n<parameter name="code">print(1)</parameter>\n</invoke>'
PROSE = "The arm is at the approach pose with the gripper open, so I'll descend and close."


def test_leading_prose_before_unfenced_code_is_dropped() -> None:
    assert _extract(f"{PROSE}\n\n{CODE}") == CODE.strip()


def test_several_leading_prose_paragraphs_are_dropped() -> None:
    reply = f"The cube moved.\n\nSo I will look again, then approach.\n\n{CODE}"
    assert _extract(reply) == CODE.strip()


def test_empty_tool_call_and_invented_error_text_before_code_are_dropped() -> None:
    reply = (
        '<invoke name="noop">\n</invoke>\n\n'
        "Tool call is not supported here; reply with code, FINISH, or GIVE_UP.\n\n\n"
        f"{CODE}"
    )
    assert _extract(reply) == CODE.strip()


def test_a_syntax_error_inside_real_code_is_not_silently_skipped() -> None:
    broken = "x = foo(\ny = 3\n\nprint(x)"
    assert _extract(broken) == broken


def test_prose_containing_an_assignment_like_equals_sign_is_kept() -> None:
    reply = f"Set a = 3 first.\n\n{CODE}"
    assert _extract(reply) == reply.strip()


def test_code_with_a_plain_syntax_error_is_returned_unchanged() -> None:
    assert _extract("move_to_joints(") == "move_to_joints("


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("FINISH", "FINISH"),
        ("finish.", "FINISH"),
        ("GIVE_UP!", "GIVE_UP"),
        (
            "The cube is held and lifted clear of the table, so the task looks complete.\n\nFINISH",
            "FINISH",
        ),
        ("I could not find the cube.\nIt may be out of view.\nGIVE_UP", "GIVE_UP"),
        (f"{CODE}\nFINISH", None),
        (f"{_BLOCK}\nFINISH", None),
        ("The task looks complete, I think.", None),
        (CODE, None),
    ],
)
def test_control_word_detection(reply: str, expected: str | None) -> None:
    assert _control_word(reply) == expected


def test_code_directly_inside_the_invoke_tag_without_a_parameter_tag() -> None:
    reply = f'<invoke name="python">\n{CODE}</invoke>\n'
    assert _extract(reply) == CODE.strip("\n")


def test_empty_blocks_after_the_first_are_ignored() -> None:
    reply = (
        f'<invoke name="python">\n{CODE}</invoke>\n\n'
        '<invoke name="python">\n</invoke>\n\n<invoke name="python">\n</invoke>\n'
    )
    assert _extract(reply) == CODE.strip("\n")
