import subprocess
from deployment import traffic_capture


def test_capture_calls_tcpdump(monkeypatch):
    calls = []

    def fake_run(cmd, *args, **kwargs):
        calls.append(cmd)

    monkeypatch.setattr('subprocess.run', fake_run)

    traffic_capture.capture()

    assert calls, 'subprocess.run should have been called'
    assert isinstance(calls[0], (list, tuple))
    assert any('tcpdump' in str(part) for part in calls[0])
