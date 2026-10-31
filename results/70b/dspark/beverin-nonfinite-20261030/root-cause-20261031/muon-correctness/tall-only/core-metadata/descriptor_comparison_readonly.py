targets=[{'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173742', 'kind': 'cpu', 'address': 22698485469120, 'matches': []}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173742.gpu', 'kind': 'gpu', 'address': 22698485469120, 'matches': [{'file_offset': 8668305520, 'vaddr': 22698484826112, 'filesz': 9564160, 'memsz': 9564160, 'flags': 3}]}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173743', 'kind': 'cpu', 'address': 22984474087360, 'matches': []}, {'path': '/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518/core_nid002488_173743.gpu', 'kind': 'gpu', 'address': 22984474087360, 'matches': [{'file_offset': 8668305520, 'vaddr': 22984473444352, 'filesz': 9564160, 'memsz': 9564160, 'flags': 3}]}]
prior=355228

import os,base64,hashlib,json,datetime
used=prior
for target in targets:
 if len(target['matches'])!=1:
  target['result']='no unique saved PT_LOAD mapping';continue
 segment=target['matches'][0];offset=segment['file_offset']+target['address']-segment['vaddr']
 assert used+64<=4194304
 fd=os.open(target['path'],os.O_RDONLY)
 try:b=os.pread(fd,64,offset)
 finally:os.close(fd)
 used+=len(b);target.update({'file_offset':offset,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode(),'result':'read64B metadata'})
print(json.dumps({'utc':datetime.datetime.utcnow().isoformat()+'Z','core_bytes_read_total':used,'core_bytes_read_this_phase':used-prior,'targets':targets}))