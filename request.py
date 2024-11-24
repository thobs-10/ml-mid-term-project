import requests
					
input_data = {
    "temperature":22,
    "hour": 4,
    "solar_radiation":0.987,
    "seasons":"Winter",
    "dew_point_temperature":12,
    "functioning_day":"Yes",
    "rainfall":76
}

response = requests.post(
    url='http://127.0.0.1:8000/predict',
    json=input_data,
    headers={'Content-Type': 'application/json'}
)
print(response.status_code)
print(response.json())