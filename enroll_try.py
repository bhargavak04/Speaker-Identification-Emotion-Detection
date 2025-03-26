import requests

url = "http://localhost:8000/enroll-speaker/"
files = {'audio': open(r'C:\Users\bharg\Downloads\SID&ED\Audio_Formatted\Yaswanth\Bode_yaswanth_kumar_Angry.mp3', 'rb')}
params = {'name': 'Yaswanth'}  # This should be sent as query parameters

response = requests.post(url, files=files, params=params)
print(response.json())