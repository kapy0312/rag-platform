import httpx

API_KEY = "AIzaSyAj9E8Qs0mxAWItS0PDi4QyHm93KdD80I8"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={API_KEY}"

resp = httpx.post(URL, json={
    "contents": [{"parts": [{"text": "你好，請用一句話自我介紹"}]}]
})
print(resp.status_code)
print(resp.json())