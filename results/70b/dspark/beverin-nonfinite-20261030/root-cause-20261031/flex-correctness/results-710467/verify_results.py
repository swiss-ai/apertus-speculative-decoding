"""Recheck raw-byte provenance and all recorded attention oracle criteria."""
from pathlib import Path
import hashlib,json,math
root=Path(__file__).resolve().parent
collection=json.loads((root/'collection.json').read_text())
for item in collection['files']:
    value=(root/'raw'/item['path']).read_bytes()
    assert len(value)==item['bytes'] and hashlib.sha256(value).hexdigest()==item['sha256']
deployment=root/'raw/deployments/710467'
expected_images={'upstream':'b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f','alps':'416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928'}
expected_cases={(p,f) for p in (False,True) for f in ('uniform','random-zero-v','saturated-constant-v','small-fp32-math-reference')}
report={'job':'710467','raw_files_verified':len(collection['files']),'raw_bytes':sum(x['bytes'] for x in collection['files']),'cells':{}}
for name,image in expected_images.items():
    cell=deployment/name
    assert (cell/'image.sha256').read_text().split()[0]==image
    assert int((cell/'exit-code.txt').read_text())==0
    result=json.loads((cell/'result.json').read_text())
    assert result['status']=='passed' and len(result['cases'])==8
    assert {(x['padded_anchors'],x['fixture']) for x in result['cases']}==expected_cases
    for case in result['cases']:
        assert all(x==0 for x in case['nonfinite'].values())
        if case['fixture']=='small-fp32-math-reference':
            assert all(x==0 for x in case['reference_nonfinite'].values())
            assert all(math.isfinite(x) and x<=.03 for x in case['relative_l2_error'].values())
        else:
            assert math.isfinite(case['max_output_error']) and case['max_output_error']<=.02
            assert math.isfinite(case['dv_mass_max_relative_error']) and case['dv_mass_max_relative_error']<=.02
    report['cells'][name]={'cases':8,'nonfinite':0,'torch':result['torch'],'torch_git':result['torch_git'],'hip':result['hip'],'image_sha256':image,'mask_sha256':result['mask_sha256'],'max_production_output_error':max(x['max_output_error'] for x in result['cases'] if 'max_output_error' in x),'max_dv_mass_relative_error':max(x['dv_mass_max_relative_error'] for x in result['cases'] if 'dv_mass_max_relative_error' in x),'max_small_reference_relative_l2_error':max(y for x in result['cases'] if 'relative_l2_error' in x for y in x['relative_l2_error'].values())}
assert (deployment/'REVISION').read_text().strip()=='e9da7e874f1f9db3ad6957d836ca94bc521657f5'
assert (deployment/'job-exit-code.txt').read_text().strip()=='0'
summary=json.loads((deployment/'summary.json').read_text())
assert all(x['passed'] and x['case_count']==8 and x['exit_code']==0 for x in summary.values())
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
