from videocad_onshape.action_codec import ActionCodec, RawModelAction, StepContext
from videocad_onshape.calibration import CanvasCalibration, ScreenRect


def test_move_action_maps_to_canvas_coordinates():
    calibration = CanvasCalibration(
        window=ScreenRect(0, 0, 1600, 1000),
        canvas=ScreenRect(100, 200, 800, 600),
        source="test",
    )
    action = ActionCodec().decode(
        RawModelAction(command=0, params=[515, 500, -1, -1, -1, -1]),
        calibration,
        StepContext(step_index=0, op="draw_rectangle"),
    )

    assert action.kind == "move"
    assert action.point == (500.0, 500.0)


def test_type_action_prefers_numeric_payload_from_step():
    calibration = CanvasCalibration(
        window=ScreenRect(0, 0, 1600, 1000),
        canvas=ScreenRect(100, 200, 800, 600),
        source="test",
    )
    action = ActionCodec().decode(
        RawModelAction(command=3, params=[-1, -1, -1, -1, -1, 612]),
        calibration,
        StepContext(step_index=1, op="add_dimension", pending_numeric_text="12.5"),
    )

    assert action.kind == "type"
    assert action.text == "12.5"

