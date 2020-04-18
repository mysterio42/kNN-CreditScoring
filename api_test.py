import requests

url = 'http://0.0.0.0:5000/classify'

payload = {
    'income': 27845.8008938469,
    'age': 55.4968525394797,
    'loan': 10871.1867897838
}

if __name__ == '__main__':
    r = requests.post(url=url, json=payload)
    print(r.text)
