"""Small daily-use check: normal EOS, complete code, long input and conversation."""
import json, os, sys, time, urllib.request
from pathlib import Path

root = Path(sys.argv[1])
results = root/'results'
results.mkdir(parents=True, exist_ok=True)
base = os.environ.get('BENCHMARK_URL', 'http://127.0.0.1:18081')

def wait():
    start = time.monotonic()
    while (root/'hold').exists():
        if time.monotonic()-start > 180:
            raise RuntimeError('guard hold timeout')
        time.sleep(.25)

def metrics(name):
    with urllib.request.urlopen(base+'/metrics', timeout=10) as response:
        data = response.read()
    (results/(name+'-metrics.txt')).write_bytes(data)

def run(name, messages, maximum):
    wait()
    body = dict(model='qwen27-test', messages=messages, max_tokens=maximum,
                temperature=0, stream=True, stream_options={'include_usage': True},
                chat_template_kwargs={'enable_thinking': False})
    (results/(name+'-request.json')).write_text(json.dumps(body, indent=2)+'\n')
    req = urllib.request.Request(base+'/v1/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type':'application/json'})
    epoch = time.time(); begin = time.perf_counter(); chunks = []; usage = None; reason = None
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b'data: '): continue
            data = line[6:].strip()
            if data == b'[DONE]': break
            row = json.loads(data)
            if row.get('usage'): usage = row['usage']
            for choice in row.get('choices', []):
                if choice.get('finish_reason'): reason = choice['finish_reason']
                content = choice.get('delta', {}).get('content') or ''
                if content: chunks.append([time.perf_counter()-begin, content])
    elapsed = time.perf_counter()-begin
    holds = any(e.get('event') == 'generation_hold' and e.get('epoch', 0) >= epoch
                for e in (json.loads(line) for line in ((root/'events.jsonl').read_text().splitlines() if (root/'events.jsonl').exists() else [])))
    n = usage['completion_tokens']; first = chunks[0][0]; last = chunks[-1][0]
    answer = ''.join(c[1] for c in chunks)
    result = dict(case=name, usage=usage, elapsed_s=elapsed, ttft_s=first,
                  decode_tokens_per_s=(n-1)/(last-first) if last>first else None,
                  end_to_end_tokens_per_s=n/elapsed, finish_reason=reason,
                  held_during_reply=holds, text=answer, chunks=chunks)
    (results/(name+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('text','chunks')}), flush=True)
    metrics(name)
    return answer, result

metrics('quality_initial')
code_prompt = ('Write a complete Python function fibonacci(n) that returns a list of the first n Fibonacci numbers, '
               'starting with 0, 1. Accept non-negative integers only. Reject bool and other types with TypeError; '
               'reject negative integers with ValueError. Do not use recursion. Return only one fenced Python code block '
               'containing the function. No imports or test execution are needed.')
code, code_result = run('complete_code', [{'role':'user','content':code_prompt}], 1024)

records = [f'Record {i:03d}: Station Cedar-{i:03d} measured {10+i%17} millimetres of rain. The water tank held {100+i*3} litres after inspection. The operator checked the filter and reported normal flow through both outlet pipes.' for i in range(1, 71)]
prompt = ('Use only these station records.\n'+'\n'.join(records)+
          '\nReturn only a JSON object with record_017_tank_litres and record_053_rain_mm. Do not include Markdown.')
# Bound the input to leave room for two conversation turns in the 4096 context.
for _ in range(8):
    req = urllib.request.Request(base+'/tokenize', data=json.dumps({'model':'qwen27-test','prompt':prompt}).encode(), headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=20) as response: tokenized = json.load(response)
    count = tokenized.get('count',len(tokenized.get('tokens',[])))
    if 2700 <= count <= 3150: break
    n = max(53, int(len(records)*2950/count)); records = records[:n]
    prompt = ('Use only these station records.\n'+'\n'.join(records)+
              '\nReturn only a JSON object with record_017_tank_litres and record_053_rain_mm. Do not include Markdown.')
else: raise RuntimeError('Long quality input outside token budget')
messages = [{'role':'user','content':prompt}]
first, _ = run('facts_first', messages, 128)
repeat, _ = run('facts_repeat', messages, 128)
messages += [{'role':'assistant','content':first}, {'role':'user','content':'Using those same records, return only a JSON object with record_053_tank_litres.'}]
followup, _ = run('conversation_1', messages, 128)
messages += [{'role':'assistant','content':followup}, {'role':'user','content':'Now return only a JSON object with record_017_rain_mm. Use the original records.'}]
last, _ = run('conversation_2', messages, 128)
# MTP drops one aligned block; use >3200 input tokens to test reuse at block1600.
boundary_n = 70
for _ in range(8):
    boundary_records = [f'Record {i:03d}: Station Cedar-{i:03d} measured {10+i%17} millimetres of rain. The water tank held {100+i*3} litres after inspection. The operator checked the filter and reported normal flow through both outlet pipes.' for i in range(1, boundary_n+1)]
    boundary_prompt = 'Use only these station records.\n'+'\n'.join(boundary_records)+'\nReturn only a JSON object with record_017_tank_litres and record_053_rain_mm. Do not include Markdown.'
    req = urllib.request.Request(base+'/tokenize',data=json.dumps({'model':'qwen27-test','prompt':boundary_prompt}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as response: tokenized=json.load(response)
    count=tokenized.get('count',len(tokenized.get('tokens',[])))
    if 3500 <= count <= 3700: break
    boundary_n=max(53,int(boundary_n*3600/count))
else: raise RuntimeError('Prefix-boundary input outside token budget')
boundary_messages=[{'role':'user','content':boundary_prompt}]
boundary_first,_=run('prefix_boundary_first',boundary_messages,128)
boundary_repeat,_=run('prefix_boundary_repeat',boundary_messages,128)
checks = {}
for name, answer, expected in [
    ('facts_first',first,{'record_017_tank_litres':151,'record_053_rain_mm':12}),
    ('facts_repeat',repeat,{'record_017_tank_litres':151,'record_053_rain_mm':12}),
    ('conversation_1',followup,{'record_053_tank_litres':259}),
    ('conversation_2',last,{'record_017_rain_mm':10}),
    ('prefix_boundary_first',boundary_first,{'record_017_tank_litres':151,'record_053_rain_mm':12}),
    ('prefix_boundary_repeat',boundary_repeat,{'record_017_tank_litres':151,'record_053_rain_mm':12})]:
    try: actual = json.loads(answer); checks[name] = {'passed':actual==expected,'actual':actual,'expected':expected}
    except ValueError: checks[name] = {'passed':False,'error':'not plain JSON','expected':expected}
checks['complete_code_natural_stop'] = code_result['finish_reason'] == 'stop'
(results/'quality-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'quality_checks':checks}),flush=True)
(results/'quality-complete.json').write_text(json.dumps({'completed':True})+'\n')
