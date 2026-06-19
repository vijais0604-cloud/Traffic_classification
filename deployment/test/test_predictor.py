import os
from deployment import predictor


def test_predictor_posts(monkeypatch, tmp_path):
    called = {}

    class DummyResp:
        def json(self):
            return {"attacks": [{"prediction": "DDoS", "src_ip": "1.1.1.1", "dst_ip": "2.2.2.2"}]}

    def fake_post(url, files=None):
        called['url'] = url
        return DummyResp()

    monkeypatch.setattr('requests.post', fake_post)

    # create a minimal flows.csv expected by predictor
    with open('flows.csv', 'w') as f:
        f.write('src_ip,dst_ip\n1.1.1.1,2.2.2.2\n')

    result = predictor.predict()
    assert 'attacks' in result
    # clean up
    os.remove('flows.csv')
