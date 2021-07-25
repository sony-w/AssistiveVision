import requests

url = 'http://127.0.0.1:5000/invocations'
my_img = {'image': open('test.jpeg', 'rb')}
payload = {'question': 'how many people are there?'}
r = requests.post(url, files=my_img, data=payload)

# convert server response into JSON format.
print(r.json())