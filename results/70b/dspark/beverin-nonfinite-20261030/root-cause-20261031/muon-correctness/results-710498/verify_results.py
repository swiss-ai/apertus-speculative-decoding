"""Verify wide Muon raw provenance and recorded native-function comparisons."""
from pathlib import Path
import hashlib,json,math
root=Path(__file__).resolve().parent
collection=json.loads((root/'collection.json').read_text())
for item in collection['files']:
    raw=(root/'raw'/item['path']).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
deployment=root/'raw/deployments/710498'
images={'upstream':'b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f','alps':'416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928'}
fixtures=('zero','tiny-random','random','rank-skewed-random','rank-one-ones','rank-one-signed')
expected={(shape,fixture) for shape in ((16,80),(8192,40960)) for fixture in fixtures}
report={'job':'710498','wide_only':True,'raw_files_verified':len(collection['files']),'raw_bytes':sum(x['bytes'] for x in collection['files']),'cells':{}}
for name,image in images.items():
    cell=deployment/name
    assert (cell/'image.sha256').read_text().split()[0]==image
    assert (cell/'exit-code.txt').read_text().strip()=='0'
    ranks=[json.loads((cell/('rank%d.json'%r)).read_text()) for r in (0,1)]
    for rank,result in enumerate(ranks):
        assert result['rank']==rank and result['world_size']==2 and result['status']=='passed'
        assert result['production_shard_flip'] and result['probe_sha256']=='bfd9286b827baaa44cae17cf6e42a976507b6934f880b37c453ed92f70e6ee63'
        assert len(result['cases'])==12
        assert {(tuple(c['shape']),c['fixture']) for c in result['cases']}==expected
        for case in result['cases']:
            assert case['status']=='passed' and case['input_mismatches']==0 and case['input_nonfinite']==0
            assert all(x==0 for x in case['nonfinite'].values())
            assert math.isfinite(case['relative_l2_error']) and case['relative_l2_error']<=.05
            assert case['norm_relative_error']<=.02 and case['clamp_relative_error']<=.02
            assert case['norm_after_clamp']['placements']==['Replicate()']
            if case['fixture']=='zero':assert case['actual_nonzero']==0
    cases=[c for r in ranks for c in r['cases']]
    report['cells'][name]={'rank_count':2,'rank_cases':len(cases),'torch':ranks[0]['torch'],'torch_git':ranks[0]['torch_git'],'hip':ranks[0]['hip'],'function_sha256':ranks[0]['function_sha256'],'native_sources':ranks[0]['native_sources'],'nonfinite':0,'max_relative_l2_error':max(c['relative_l2_error'] for c in cases),'max_norm_relative_error':max(c['norm_relative_error'] for c in cases),'rank_one_relative_l2_errors':[c['relative_l2_error'] for c in cases if c['fixture'].startswith('rank-one')],'shapes':[list(shape) for shape in sorted({tuple(c['shape']) for c in cases})]}
assert (deployment/'REVISION').read_text().strip()=='cdf9adb5c1c511b1aad5a1c9b306c2ba30b22563'
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
