targets=[{'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173742', 'offset': 74548, 'bytes': 73403, 'name': 'CORE', 'type': 1179208773}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173742.gpu', 'offset': 17612, 'bytes': 920, 'name': 'AMDGPU', 'type': 33}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173743', 'offset': 74548, 'bytes': 73403, 'name': 'CORE', 'type': 1179208773}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173743.gpu', 'offset': 17612, 'bytes': 920, 'name': 'AMDGPU', 'type': 33}]
previous_bytes=183462

import os,pathlib,hashlib,json,base64,sys,subprocess,datetime
used=previous_bytes;out=[]
for t in targets:
 assert used+t['bytes']<=4194304
 fd=os.open(t['path'],os.O_RDONLY)
 try:b=os.pread(fd,t['bytes'],t['offset'])
 finally:os.close(fd)
 used+=len(b);assert len(b)==t['bytes'];t.update({'data':base64.b64encode(b).decode(),'sha256':hashlib.sha256(b).hexdigest()});out.append(t)
versions={}
for name in ['readelf','rocgdb']:
 a=subprocess.run([name,'--version'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
 versions[name]={'returncode':a.returncode,'output':a.stdout.decode(errors='replace')[:1200],'stderr':a.stderr.decode(errors='replace')[:600]}
image='/capstor/scratch/cscs/faruk_zahiragic/container-images/vllm-openai-rocm-v0.31.0.sqsh'
a=subprocess.run(['unsquashfs','-cat',image,'usr/include/linux/kfd_ioctl.h'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
header={'image':image,'path':'usr/include/linux/kfd_ioctl.h','returncode':a.returncode,'stderr':a.stderr.decode(errors='replace')[:1000]}
if a.returncode==0:header.update({'data':base64.b64encode(a.stdout).decode(),'sha256':hashlib.sha256(a.stdout).hexdigest(),'bytes':len(a.stdout)})
print(json.dumps({'utc':datetime.datetime.utcnow().isoformat()+'Z','core_bytes_read_this_phase':used-previous_bytes,'core_bytes_read_total':used,'notes':out,'tool_versions':versions,'kfd_header':header}))