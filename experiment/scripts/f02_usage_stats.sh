#!/bin/bash
# F-02: usage 统计验证 - 1A1F eager
set -e
MODEL_PATH="/models/DeepSeek-V2-Lite"
AFD_PORT=${AFD_PORT:-6240}
API_PORT=${API_PORT:-18010}
LOG_DIR="/workspace/afd-plugin/experiment/logs"

echo "=== F-02: usage 统计验证 ==="

# 启动 AFD (复用 start_afd 逻辑)
bash /workspace/afd-plugin/experiment/scripts/start_afd.sh 1a1f eager 0 1 --trust-remote-code
# 注意 start_afd.sh 已启动服务, 此处直接测试

# 测试1: 验证 usage 统计
RESPONSE=$(curl -s http://127.0.0.1:$API_PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v2-lite-afd-attention","prompt":"The capital of France is","max_tokens":8,"temperature":0}')

echo "$RESPONSE" | python3 -c "
import sys, json
resp = json.load(sys.stdin)
usage = resp.get('usage', {})
assert usage.get('prompt_tokens', 0) > 0, f'prompt_tokens should be > 0, got {usage}'
assert usage.get('completion_tokens', 0) > 0, f'completion_tokens should be > 0, got {usage}'
assert usage.get('total_tokens', 0) == usage['prompt_tokens'] + usage['completion_tokens'], 'total != prompt + completion'
print(f'PASS: prompt_tokens={usage[\"prompt_tokens\"]}, completion_tokens={usage[\"completion_tokens\"]}, total={usage[\"total_tokens\"]}')
"

# 测试2: 验证错误模型名返回错误
echo "Test invalid model name..."
curl -s http://127.0.0.1:$API_PORT/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nonexistent-model","prompt":"test","max_tokens":1}' | python3 -c "
import sys, json
resp = json.load(sys.stdin)
assert 'error' in resp or 'detail' in resp, 'Expected error for invalid model'
print('PASS: invalid model name returns error')
"

# 测试3: 多请求并发
echo "Test concurrent requests..."
python3 -c "
import urllib.request, json, threading, time
results = []
errors = []
def send_request(i):
    try:
        payload = json.dumps({
            'model': 'deepseek-v2-lite-afd-attention',
            'prompt': f'Hello world {i}',
            'max_tokens': 8,
            'temperature': 0
        }).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:$API_PORT/v1/completions',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            results.append(data)
    except Exception as e:
        errors.append(str(e))

threads = [threading.Thread(target=send_request, args=(i,)) for i in range(4)]
for t in threads: t.start()
for t in threads: t.join(timeout=60)

assert len(errors) == 0, f'Errors in concurrent requests: {errors}'
assert len(results) == 4, f'Expected 4 results, got {len(results)}'
for r in results:
    assert 'choices' in r and len(r['choices']) > 0
    assert len(r['choices'][0]['text']) > 0
print(f'PASS: 4 concurrent requests all succeeded')
"

echo "F-02: PASS"

# 清理
bash /workspace/afd-plugin/experiment/scripts/stop_afd.sh 1a1f_eager
