#!/usr/bin/env python3
"""C128 boundary probe: real requests and logs only; does not modify SGLang."""
import json,time,uuid,urllib.request,urllib.error,re,sys,traceback
from pathlib import Path
RUN=Path(sys.argv[1]); OP=urllib.request.build_opener(urllib.request.ProxyHandler({}))
LOG=RUN/'server.log'; ROWS=[]
IDS=[1000+(i*37)%90000 for i in range(8500)]
SALT='c128-'+uuid.uuid4().hex
(RUN/'probe_input.json').write_text(json.dumps(dict(input_ids=IDS,cache_salt=SALT,routed_dp_rank=0)))
def save(name,obj): (RUN/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def http(path,data=None,timeout=600):
 req=urllib.request.Request('http://127.0.0.1:8001/'+path,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'})
 with OP.open(req,timeout=timeout) as r:return r.status,r.read().decode()
def metrics():
 s=OP.open('http://127.0.0.1:9004/metrics',timeout=5).read().decode()
 return {a[0]:float(a[1]) for l in s.splitlines() if not l.startswith('#') and len(a:=l.split())==2 and 'put_' in a[0]}
def settle(label,before):
 prev=None; stable=0; samples=[]
 for _ in range(60):
  m=metrics();samples.append(m)
  stable=stable+1 if m==prev else 0;prev=m
  ended=sum(m.get(k,0)-before.get(k,0) for k in ['master_put_end_requests_total','master_batch_put_end_items_total'])
  if stable>=5 and ended>0:break
  time.sleep(2)
 save(label+'_backup_metrics.json',samples)
 print('backup',label,'completed_delta',ended,'stable_samples',stable,flush=True)
 if ended<=0:raise RuntimeError('No successful-put completion evidence after seed')
 if any(m.get(k,0)>before.get(k,0) for k in m if 'fail' in k):raise RuntimeError('Mooncake put failure increased')
def flush(label):
 for i in range(30):
  try:
   status,body=http('flush_cache',{},60);save(label+'_flush.json',dict(status=status,body=body));return
  except urllib.error.HTTPError as e:
   if e.code!=400:raise
   time.sleep(1)
 raise RuntimeError('flush stayed busy')
def request(label,n,salt=SALT):
 rid=label+'-'+uuid.uuid4().hex[:8]
 payload=dict(input_ids=IDS[:n],rid=rid,cache_salt=salt,routed_dp_rank=0,sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True))
 save(label+'_request.json',payload);offset=LOG.stat().st_size;t=time.monotonic()
 status,body=http('generate',payload);response=json.loads(body);save(label+'_response.json',response)
 time.sleep(.5)
 with LOG.open('rb') as f:f.seek(offset);logs=f.read().decode(errors='replace')
 (RUN/(label+'_server.log')).write_text(logs)
 diag=[l for l in logs.splitlines() if 'C128_PREFETCH_DIAG' in l]
 (RUN/(label+'_diag.log')).write_text('\n'.join(diag)+'\n')
 row=dict(label=label,rid=rid,input_tokens=n,seconds=time.monotonic()-t,meta_info=response.get('meta_info'),diagnostics=diag)
 ROWS.append(row);save('results.json',ROWS)
 print(label,'input',n,'meta',json.dumps(row['meta_info']),flush=True)
 for l in diag:
  if any(x in l for x in ['phase=query','phase=result','phase=publish','phase=match_validator']):print(l,flush=True)
 return any('phase=publish' in l and 'unaligned=True' in l for l in diag)
def main():
 for _ in range(240):
  try:
   if http('ready',timeout=5)[0]==200:break
  except Exception:pass
  time.sleep(5)
 else:raise RuntimeError('readiness timeout')
 save('server_info.json',json.loads(http('get_server_info')[1]))
 flush('before_seed');before=metrics();request('seed_long',8193);settle('seed_long',before)
 for boundary in [4096,4224,4480,6016,6144]:
  label='root_'+str(boundary);flush(label)
  unaligned=request(label,boundary+1)
  if unaligned:request(label+'_extend',boundary+257)
 flush('prefix_case');request('prefix_2048',2049);request('prefix_to_4480',4481);request('prefix_to_6144',6145)
 for boundary in [4224,4480,6016]:
  salt=SALT+'-exact-'+str(boundary);label='exact_'+str(boundary)
  flush(label+'_seed');before=metrics();request(label+'_seed',boundary+1,salt);settle(label,before)
  flush(label+'_l3');unaligned=request(label+'_l3',boundary+1,salt)
  if unaligned:request(label+'_extend',boundary+257,salt)
 print('DONE',str(RUN),flush=True)
try:main()
except Exception as e:
 save('error.json',dict(error=repr(e),traceback=traceback.format_exc()));traceback.print_exc();sys.exit(1)
