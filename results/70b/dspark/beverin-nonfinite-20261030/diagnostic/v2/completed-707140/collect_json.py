#!/usr/bin/env python3
"""Read-only explicit Beverin collection using one compressed JSON response."""
import ast,base64,hashlib,importlib.util,json,pathlib,subprocess,zlib
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("watch",str(root.parent/"watch_diagnostic.py"));watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
source=ast.parse((root/"collect_readonly.py").read_text())
remote=next(ast.literal_eval(node.value) for node in source.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REMOTE" for t in node.targets))
remote=remote[:remote.index("path=run/'checkpoints/0/model.safetensors'")]+r'''
import zlib,base64
result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'run':str(run),'job_id':job,'remote_mutations':False,'files':{}}
for name,p in sorted(files.items()):
 raw=p.read_bytes();result['files'][name]={'cluster_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'zlib_base64':base64.b64encode(zlib.compress(raw,1)).decode()}
extras={'sacct.txt':accounting.encode(),'terminal.json':json.dumps({'failure_marker_exists':(run/'checkpoints/numerical-failure.json').exists(),'capture_files':[str(p) for p in (run/'checkpoints/numerical-failure-batches').glob('*.pt')],'final_training_state':json.loads((run/'checkpoints/0/training_state.json').read_text())}).encode()}
for name,raw in extras.items():result['files'][name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'zlib_base64':base64.b64encode(zlib.compress(raw,1)).decode()}
print(json.dumps(result))
'''
response=subprocess.run(watch.SSH,input=remote,text=True,capture_output=True,timeout=120)
if response.returncode:raise RuntimeError(response.stderr)
archive=pathlib.Path('/tmp/beverin-nan-707140-completed-20261031-files-only.json')
with archive.open('x') as stream:stream.write(response.stdout)
record=json.loads(response.stdout);rawdir=root/'raw';rawdir.mkdir(exist_ok=False);manifest=[]
for name,item in record['files'].items():
 raw=zlib.decompress(base64.b64decode(item.pop('zlib_base64')));assert hashlib.sha256(raw).hexdigest()==item['sha256'];p=rawdir/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);manifest.append(dict(item,archive_path=name))
record['files']=manifest;(rawdir/'collection.json').write_text(json.dumps(record,indent=2)+'\n');(rawdir/'SOURCE_SHA256SUMS').write_text(''.join(x['sha256']+'  '+x['archive_path']+'\n' for x in manifest))
print(json.dumps({'files':len(manifest),'raw_bytes':sum(x['bytes'] for x in manifest),'compressed_response_bytes':len(response.stdout),'all_hashes_verified':True}))
