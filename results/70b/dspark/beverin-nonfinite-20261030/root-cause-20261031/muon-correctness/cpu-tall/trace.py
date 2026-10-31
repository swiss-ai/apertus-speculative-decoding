"""Tiny CPU placement trace; manually mirrors the pinned Muon loop, no training."""
import datetime,json,os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor,Shard,_collective_utils,placement_types

def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo',timeout=datetime.timedelta(seconds=60))
    rank=dist.get_rank();world=dist.get_world_size();mesh=init_device_mesh('cpu',(world,))
    def production_flip(value,gather_dim,shard_dim,device_mesh,mesh_dim):
        gathered=funcol.all_gather_tensor(value,gather_dim,(device_mesh,mesh_dim))
        if isinstance(gathered,funcol.AsyncCollectiveTensor):gathered=gathered.wait()
        return torch.chunk(gathered,device_mesh.size(mesh_dim),dim=shard_dim)[device_mesh.get_local_rank(mesh_dim)].contiguous()
    _collective_utils.shard_dim_alltoall=production_flip
    placement_types.shard_dim_alltoall=production_flip
    out=Path(os.environ['TRACE_OUTPUT']);out.mkdir(exist_ok=True,parents=True)
    report={'rank':rank,'torch':torch.__version__,'host_dispatch_only':True,'shape':[80,16],'traces':[]}
    for fixture in ['rank-one-ones','random']:
        torch.manual_seed(20261031)
        dense=torch.ones((80,16),dtype=torch.bfloat16) if fixture=='rank-one-ones' else torch.randn((80,16),dtype=torch.bfloat16)
        sharded=DTensor.from_local(dense.chunk(world,0)[rank].clone(),device_mesh=mesh,placements=[Shard(0)],run_check=False)
        def compare(name,value,reference,step=None):
            full=value.full_tensor()
            diff=full.float()-reference.float()
            err=float(diff.norm())/max(float(reference.float().norm()),1e-30)
            item={'fixture':fixture,'step':step,'operation':name,'placements':[repr(p) for p in value.placements],
                  'relative_l2_error':err,'nonfinite':int((~torch.isfinite(full)).sum())}
            report['traces'].append(item)
            assert item['nonfinite']==0
            return item
        g=sharded.T;ref=dense.T.clone();compare('transpose',g,ref)
        g.div_(g.norm().clamp(min=1e-7));ref.div_(ref.norm().clamp(min=1e-7));compare('normalize',g,ref)
        for step in range(5):
            gram=g@g.T;refgram=ref@ref.T;compare('gram',gram,refgram,step)
            update=torch.addmm(gram,gram,gram,beta=-4.7750,alpha=2.0315)
            refupdate=torch.addmm(refgram,refgram,refgram,beta=-4.7750,alpha=2.0315)
            compare('gram-update-addmm',update,refupdate,step)
            g=torch.addmm(g,update,g,beta=3.4445)
            ref=torch.addmm(ref,refupdate,ref,beta=3.4445);compare('ortho-update-addmm',g,ref,step)
        result=compare('final-transpose',g.T,ref.T)
        assert result['relative_l2_error']<=.05
    report['status']='passed'
    (out/('trace-rank%d.json'%rank)).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'rank':rank,'status':'passed','traces':len(report['traces'])}),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
