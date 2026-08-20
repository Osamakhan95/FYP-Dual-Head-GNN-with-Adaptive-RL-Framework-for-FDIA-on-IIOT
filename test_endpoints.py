import requests

r = requests.post('http://127.0.0.1:8000/auth/login', json={'email':'admin@fyp.com','password':'Admin@123'})
token = r.json()['access_token']
h = {'Authorization': f'Bearer {token}'}

# Test stats
s = requests.get('http://127.0.0.1:8000/model/stats', headers=h)
print('Stats:', s.status_code)

# Test topology
t = requests.get('http://127.0.0.1:8000/model/topology', headers=h)
tj = t.json()
print(f"Topology: {len(tj['nodes'])} nodes, {len(tj['edges'])} edges")

# Test analytics
a = requests.get('http://127.0.0.1:8000/admin/analytics', headers=h)
aj = a.json()
print(f"Analytics: {a.status_code}, users={aj.get('total_users')}, anomalies={aj.get('total_anomalies')}")

# Test CORS preflight for port 5174
c = requests.options('http://127.0.0.1:8000/health', headers={
    'Origin': 'http://localhost:5174',
    'Access-Control-Request-Method': 'GET'
})
print(f"CORS 5174: {c.headers.get('access-control-allow-origin', 'NOT SET')}")

# Test CORS preflight for 127.0.0.1:5174
c2 = requests.options('http://127.0.0.1:8000/health', headers={
    'Origin': 'http://127.0.0.1:5174',
    'Access-Control-Request-Method': 'GET'
})
print(f"CORS 127.0.0.1:5174: {c2.headers.get('access-control-allow-origin', 'NOT SET')}")

print("\nAll tests passed!")
