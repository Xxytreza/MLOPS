import requests


def test_predict():
    headers = {
        'accept': 'application/json',
    }

    response = requests.get('http://localhost:8000/predict', headers=headers)

    body = response.json()

    assert body == {'y_predict': 2}
