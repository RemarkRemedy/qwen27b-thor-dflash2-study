import json, os, sys, time, urllib.request
from pathlib import Path
root=Path(sys.argv[1]); results=root/'results'; results.mkdir(parents=True, exist_ok=True)
base=os.environ.get('BENCHMARK_URL', 'http://127.0.0.1:18081')
def wait():
    start=time.monotonic()
    while (root/'hold').exists():
        if time.monotonic()-start>180: raise RuntimeError('guard hold timeout')
        time.sleep(.25)
def make_prompt(n):
    records = [f"Record {i:03d}: Station Cedar-{i:03d} measured {10+i%17} millimetres of rain. The water tank held {100+i*3} litres after inspection. The operator checked the filter and reported normal flow through both outlet pipes." for i in range(1,n+1)]
    return "Read the following station records. Use only these records for the requested facts.\n" + "\n".join(records) + "\nStart your answer with the tank volume for Record 017 and the rain amount for Record 053. Then explain how these records could support routine water-system checks. Use clear prose."

n=70
for attempt in range(8):
    prompt=make_prompt(n)
    req=urllib.request.Request(base+'/tokenize',data=json.dumps({'model':'qwen27-test','prompt':prompt}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=20) as r: tokenization=json.load(r)
    prompt_tokens=tokenization.get('count',len(tokenization.get('tokens',[])))
    if 2800 <= prompt_tokens <= 3300: break
    n=max(53,int(n*3000/prompt_tokens))
else: raise RuntimeError('Could not create prompt within requested token budget')
(results/'long-input.json').write_text(json.dumps({'plain_text_tokens':prompt_tokens,'records':n,'expected_record_017_tank_litres':151,'expected_record_053_rain_mm':12,'prompt':prompt},indent=2)+'\n')
cases=[('warmup','Count from one to twenty, spelling out each number and giving its square.',32),('prose','Explain how rain forms and how the water cycle supports plants and rivers. Write about 400 words in clear English.',256),('code','Write a Python function that returns the first n Fibonacci numbers. Include input validation, a short explanation, and five useful tests.',256),('long_prose',prompt,256)]
base_cases=cases[1:]
cases=cases[:1]+[(name+'_r'+str(i+1),text,count) for i in range(3) for name,text,count in (base_cases[i:]+base_cases[:i])]
for name,prompt,count in cases:
    wait(); body=dict(model='qwen27-test',messages=[dict(role='user',content=prompt)],max_tokens=count,temperature=0,
      stream=True,stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':False},ignore_eos=True,return_token_ids=True)
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    start_epoch=time.time(); begin=time.perf_counter(); chunks=[]; token_chunks=[]; usage=None; holds=False
    with urllib.request.urlopen(req,timeout=240) as r:
        for line in r:
            if (root/'hold').exists(): holds=True
            if not line.startswith(b'data: '):continue
            data=line[6:].strip()
            if data==b'[DONE]':break
            row=json.loads(data)
            if row.get('usage'):usage=row['usage']
            for choice in row.get('choices',[]):
                if choice.get('token_ids'):token_chunks.append([time.perf_counter()-begin,choice['token_ids']])
                text=choice.get('delta',{}).get('content') or ''
                if text:chunks.append([time.perf_counter()-begin,text])
    elapsed=time.perf_counter()-begin
    if (root/'events.jsonl').exists():
        holds = holds or any(e.get('event') == 'generation_hold' and e.get('epoch', 0) >= start_epoch for e in (json.loads(line) for line in ((root/'events.jsonl').read_text().splitlines() if (root/'events.jsonl').exists() else [])))
    n=usage['completion_tokens']; first=chunks[0][0]; last=chunks[-1][0]
    assert sum(len(c[1]) for c in token_chunks)==n, 'Token IDs must match completion usage'
    token_first=token_chunks[0][0]; token_last=token_chunks[-1][0]
    decode_rate=sum(len(c[1]) for c in token_chunks[1:])/(token_last-token_first)
    result=dict(case=name,prompt=prompt,usage=usage,elapsed_s=elapsed,ttft_s=first,
      decode_tokens_per_s=decode_rate,
      end_to_end_tokens_per_s=n/elapsed,stream_chunks=len(chunks),text=''.join(c[1] for c in chunks),chunks=chunks,
      held_during_reply=holds,first_batch_tokens=len(token_chunks[0][1]),token_chunks=token_chunks,measurement='Exact returned tokens after first batch divided by stream interval; primary score uses full request time')
    (results/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('chunks','token_chunks','text','prompt')}),flush=True)
    with urllib.request.urlopen(base+'/metrics',timeout=5) as r:(results/(name+'-metrics.txt')).write_bytes(r.read())
    if holds and not name.endswith('_retry'):
        cases.append((name+'_retry',prompt,count))
(results/'complete.json').write_text(json.dumps({'completed':True})+'\n')
