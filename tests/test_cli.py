import json

from videocad_onshape.cli import main


class FakeApplication:
    def __init__(self, config):
        self.config = config

    def healthcheck(self):
        class Result:
            def to_dict(self):
                return {"ok": True, "items": [{"name": "predictor", "ok": True, "detail": "ready"}]}

            @property
            def ok(self):
                return True

            @property
            def items(self):
                return []

        return Result()

    def execute_prompt(self, prompt, mode):
        class Summary:
            def to_dict(self):
                return {"mode": mode, "status": "completed", "step_count": 3, "prompt": prompt}

        return Summary()

    def calibrate(self):
        return {"canvas": {"width": 100, "height": 100}}


def test_cli_healthcheck_json(monkeypatch, capsys):
    monkeypatch.setattr("videocad_onshape.cli.Application", FakeApplication)

    exit_code = main(["healthcheck", "--json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["ok"] is True


def test_cli_dry_run_prints_summary(monkeypatch, capsys):
    monkeypatch.setattr("videocad_onshape.cli.Application", FakeApplication)

    exit_code = main(["dry-run", "--prompt", "Create a centered rectangle 10 by 20 and extrude it 5."])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
