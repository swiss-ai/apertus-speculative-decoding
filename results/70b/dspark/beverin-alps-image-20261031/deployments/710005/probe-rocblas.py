import cProfile,io,pstats,json,os,time
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor,Shard,Replicate
rank=int(os.environ['RANK']); torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl')
mesh=init_device_mesh('cuda',(dist.get_world_size(),))
torch.backends.cuda.preferred_blas_library('cublas')
for placement in (None,Replicate(),Shard(0)):
    x=torch.randn(1024,1024,device='cuda',dtype=torch.bfloat16)
    if placement is not None: x=DTensor.from_local(x,mesh,[placement],run_check=False)
    for i in range(5):
        torch.cuda.synchronize();start=time.perf_counter();cpu=time.process_time()
        y=x @ x.mT
        if isinstance(y,DTensor): y.to_local()
        torch.cuda.synchronize()
        print(json.dumps({'rank':rank,'torch':torch.__version__,'placement':str(placement),'iteration':i,'wall_ms':1000*(time.perf_counter()-start),'cpu_ms':1000*(time.process_time()-cpu)}),flush=True)
profile=cProfile.Profile(); profile.enable()
y=x @ x.mT; y.to_local(); torch.cuda.synchronize()
profile.disable()
if rank == 0:
    stats=io.StringIO(); pstats.Stats(profile,stream=stats).sort_stats('cumtime').print_stats(35); print(stats.getvalue(),flush=True)
dist.destroy_process_group()
