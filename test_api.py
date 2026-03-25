import requests

url = " https://fraud-detection-api-l2t8.onrender.com"

files = {"file": open("test.wav", "rb")}

response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Raw Response:", response.text)