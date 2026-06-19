from deployment import main


def test_run_pipeline_calls_components(monkeypatch):
    calls = []

    monkeypatch.setattr(main, 'capture', lambda: calls.append('capture'))
    monkeypatch.setattr(main, 'generate_csv', lambda: calls.append('generate_csv'))

    def fake_predict():
        return {"attacks": [{"prediction": "DDoS", "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"}]}

    monkeypatch.setattr(main, 'predict', fake_predict)

    main.run_pipeline()

    assert 'capture' in calls and 'generate_csv' in calls
