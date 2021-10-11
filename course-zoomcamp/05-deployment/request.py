import requests

url = 'http://localhost:9696/predict'
customer = {"contract": "two_year", "tenure": 1, "monthlycharges": 10}
docker_customer = {"contract": "two_year", "tenure": 12, "monthlycharges": 10}

response = requests.post(url, json=docker_customer).json()
print(response)
