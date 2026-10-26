import json, os, sys, torch
n=torch.cuda.device_count()
assert n == 1, f'Expected one GPU, found {n}'
assert torch.cuda.get_device_capability(0)==(11,0), 'Expected integrated Thor SM110'
print('VERIFIED_DEVICE',json.dumps({'name':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'cuda':torch.version.cuda,'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}),flush=True)
os.execvp('vllm',['vllm','serve']+sys.argv[1:])
