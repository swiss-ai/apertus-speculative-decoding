import json, os, statistics, time
import torch
print(json.dumps({'torch':torch.__version__,'hip':torch.version.hip,'device':torch.cuda.get_device_name(0),'tunable_enabled':torch.cuda.tunable.is_enabled(),'tuning_enabled':torch.cuda.tunable.tuning_is_enabled(),'blas':str(torch.backends.cuda.preferred_blas_library())}),flush=True)
for library in ('default','cublas','cublaslt'):
    try:
        torch.backends.cuda.preferred_blas_library(library)
        for dtype in (torch.bfloat16, torch.float32):
            a=torch.randn(1024,1024,device='cuda',dtype=dtype)
            b=torch.randn_like(a)
            torch.mm(a,b); torch.cuda.synchronize()
            rows=[]
            for i in range(5):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(); t=time.perf_counter(); c=time.process_time(); start.record()
                torch.mm(a,b); end.record(); torch.cuda.synchronize()
                rows.append({'wall_ms':(time.perf_counter()-t)*1000,'cpu_ms':(time.process_time()-c)*1000,'gpu_ms':start.elapsed_time(end)})
            print(json.dumps({'library':library,'dtype':str(dtype),'samples':rows}),flush=True)
    except Exception as error:
        print(json.dumps({'library':library,'error':repr(error)}),flush=True)
