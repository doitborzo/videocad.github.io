from videocad_onshape.planner import RuleBasedPlanner
from videocad_onshape.renderer import CanonicalRenderer


def test_renderer_emits_target_image_per_step():
    planner = RuleBasedPlanner()
    program = planner.plan("Create a centered rectangle 10 by 20 and extrude it 5.")
    rendered = CanonicalRenderer().render_program_steps(program)

    assert len(rendered) == len(program.steps)
    assert all(item.image.size == (224, 224) for item in rendered)

