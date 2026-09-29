import urllib.request
import json
import time

t0 = time.time()
req = urllib.request.Request(
    'http://127.0.0.1:11434/api/chat',
    data=json.dumps({
        'model': 'qwen3:4b',
        'messages': [
            {'role': 'system', 'content': 'You are SovereignAI, a helpful assistant. Reply directly to the user in 1-2 sentences.'},
            {'role': 'user', 'content': 'Hi'}
        ],
        'stream': True
    }).encode(),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        first = True
        full = []
        for line in resp:
            if line.strip():
                chunk = json.loads(line.decode())
                txt = chunk.get('message', {}).get('content', '')
                if txt:
                    if first:
                        print(f'First token latency: {time.time() - t0:.2f}s')
                        first = False
                    full.append(txt)
        print(f'Total time: {time.time() - t0:.2f}s')
        safe_response = ''.join(full).encode('ascii', 'replace').decode('ascii')
        print('Response:', repr(safe_response))
except Exception as e:
    print('Error:', e)
