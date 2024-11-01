import requests
import json

url = 'http://127.0.0.1:8000/start_detect'

data ={"rtspUrl": "D:\ANACONDA\envs\s\\test.mp4","id": "0"}


headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=json.dumps(data), headers=headers)

print(response.json())
if response.status_code == 200:
    print('请求成功')
    print(response.text)
else:
    print('请求失败')
    print(response.text)
