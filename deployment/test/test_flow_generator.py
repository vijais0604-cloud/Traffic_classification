import subprocess
from deployment import flow_generator


def test_generate_csv_calls_cicflowmeter(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr('subprocess.run', fake_run)

    flow_generator.generate_csv()

    assert calls, 'subprocess.run should have been called'
    # first call should be a list containing 'cicflowmeter'
    assert isinstance(calls[0], (list, tuple))
    assert any('cicflowmeter' in str(part) for part in calls[0])
