import pytest

from videocad_onshape.errors import UnsupportedPromptError
from videocad_onshape.planner import RuleBasedPlanner


def test_rule_planner_builds_rectangle_program():
    planner = RuleBasedPlanner()
    program = planner.plan("Create a centered rectangle 10 by 20 on the front plane and extrude it 5.")

    assert program.steps[0].op == "new_part_studio"
    assert [step.op for step in program.steps][-1] == "extrude_add"


@pytest.mark.parametrize(
    "prompt",
    [
        "Build an assembly with two mates.",
        "Add a fillet to the edges.",
        "Import arbitrary geometry and modify it.",
        "Make a nice bracket.",
    ],
)
def test_rule_planner_rejects_unsupported_or_ambiguous_prompts(prompt):
    planner = RuleBasedPlanner()
    with pytest.raises(UnsupportedPromptError):
        planner.plan(prompt)

