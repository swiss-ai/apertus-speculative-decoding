import os,struct,pathlib,json,hashlib,base64,datetime,sys,shutil
base=pathlib.Path('/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-tall-correctness-20261031T163340Z/deployments/710518')
paths=[base/name for name in ['core_nid002488_173742','core_nid002488_173742.gpu','core_nid002488_173743','core_nid002488_173743.gpu']]
limit=1048576;used=0;ranges=[];files=[]
for p in paths:
 fd=os.open(str(p),os.O_RDONLY);st=os.fstat(fd)
 def read(offset,length,kind):
  global used
  if used+length>limit:raise RuntimeError('First-pass1MiB limit; stop before additional read')
  b=os.pread(fd,length,offset);used+=len(b)
  ranges.append({'path':str(p),'offset':offset,'requested_bytes':length,'actual_bytes':len(b),'kind':kind,'sha256':hashlib.sha256(b).hexdigest(),'data':base64.b64encode(b).decode()})
  return b
 try:
  header=read(0,64,'ELF header');assert header[:4]==b'\x7fELF'
  assert header[4]==2 and header[5]==1
  h=struct.unpack('<16sHHIQQQIHHHHHH',header);phnum=h[10]
  if phnum==65535:
   sh=read(h[6],h[11],'extended section0 header');phnum=struct.unpack('<IIQQQQIIQQ',sh[:64])[7]
  assert h[9]>=56 and h[9]<=128 and phnum<20000
  phbytes=read(h[5],h[9]*phnum,'ELF program headers')
  item={'path':str(p),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'elf_type':h[1],'machine':h[2],'entry':h[4],'phnum':phnum,'notes':[],'load_segment_count':0}
  for i in range(phnum):
   ph=struct.unpack('<IIQQQQQQ',phbytes[i*h[9]:i*h[9]+56])
   if ph[0]==1:item['load_segment_count']+=1
   if ph[0]!=4:continue
   off=ph[2];end=off+ph[5];noteindex=0
   while off+12<=end:
    nh=read(off,12,'PT_NOTE header');namesz,descsz,typ=struct.unpack('<III',nh)
    if namesz==0 and descsz==0:break
    assert namesz<4096 and off+12+((namesz+3)//4*4)+descsz<=end
    name=read(off+12,namesz,'PT_NOTE name').rstrip(b'\0').decode(errors='replace')
    desc_off=off+12+(namesz+3)//4*4
    item['notes'].append({'name':name,'type':typ,'type_hex':hex(typ),'descriptor_offset':desc_off,'descriptor_bytes':descsz})
    off=desc_off+(descsz+3)//4*4;noteindex+=1
    assert noteindex<10000
  files.append(item)
 finally:os.close(fd)
print(json.dumps({'utc':datetime.datetime.utcnow().isoformat()+'Z','job':'710518','source_revision':'0030d3123e3bf7b8fa4f6c41e3615ab57b45203c','python':sys.version,'tools':{name:shutil.which(name) for name in ['readelf','rocgdb','gdb','llvm-readelf']},'read_budget_phase1':limit,'core_bytes_read':used,'files':files,'read_ranges':ranges}))