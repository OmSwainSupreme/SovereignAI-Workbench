import urllib.request
import json
import time

ws_req = urllib.request.Request(
    'http://127.0.0.1:8000/api/v1/tasks',
    data=json.dumps({
        'task': 'Hi',
        'attachments': []
    }).encode(),
    headers={'Content-Type': 'application/json'}
)
resp = urllib.request.urlopen(ws_req)
task = json.loads(resp.read().decode())
task_id = task['id']
print(f'Started task {task_id}')

for _ in range(120):
    time.sleep(2)
    t_req = urllib.request.Request(f'http://127.0.0.1:8000/api/v1/tasks/{task_id}')
    t_resp = urllib.request.urlopen(t_req)
    t_data = json.loads(t_resp.read().decode())
    status = t_data['status']
    phase = t_data.get('phase')
    model = t_data.get('model')
    print(f'Status: {status}, Phase: {phase}, Model: {model}')
    if status in ('complete', 'failed', 'cancelled'):
        print('Final messages:')
        for m in t_data.get('messages', []):
            print(f"[{m['role']}]: {m['content']}")
        break
