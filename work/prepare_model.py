import json, hashlib, sys
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parent.parent
out=root/'outputs'
if len(sys.argv)>1 and sys.argv[1]=='full-quality': out=out/'full-quality'
ply=out/'meetingroom.ply'
with ply.open('rb') as f:
    header=[]
    while True:
        line=f.readline()
        if not line: raise ValueError('Missing PLY header terminator')
        header.append(line.decode('ascii').strip())
        if line.strip()==b'end_header': break
    assert 'format binary_little_endian 1.0' in header
    count=int(next(x for x in header if x.startswith('element vertex ')).split()[-1])
    props=[x.split() for x in header if x.startswith('property ')]
    assert all(x[1]=='float' for x in props), props
    dtype=np.dtype([(x[2],'<f4') for x in props])
    values=np.fromfile(f,dtype=dtype,count=count)
    assert len(values)==count and not f.read(1), 'PLY length mismatch'
    assert all(np.isfinite(values[name]).all() for name in dtype.names), 'Nonfinite values'
xyz=np.column_stack([values[x] for x in ['x','y','z']])
scale=np.exp(np.column_stack([values[f'scale_{i}'] for i in range(3)]))
alpha=1/(1+np.exp(-np.clip(values['opacity'],-80,80)))
rgb=np.column_stack([values[f'f_dc_{i}'] for i in range(3)])*.28209479177387814+.5
quat=np.column_stack([values[f'rot_{i}'] for i in range(4)])
norm=np.linalg.norm(quat,axis=1,keepdims=True)
assert (norm>0).all() and np.isfinite(scale).all() and (scale>0).all()
quat=quat/norm
order=np.argsort(-(scale.prod(axis=1)*alpha))
rows=np.zeros(count,dtype=[('position','<f4',(3,)),('scale','<f4',(3,)),('rgba','u1',(4,)),('rotation','u1',(4,))])
rows['position']=xyz[order]; rows['scale']=scale[order]
rows['rgba']=np.clip(np.column_stack([rgb,alpha])[order]*255,0,255).astype('u1')
rows['rotation']=np.clip(quat[order]*128+128,0,255).astype('u1')
assert rows.dtype.itemsize==32
rows.tofile(out/'meetingroom.splat')
assert (out/'meetingroom.splat').stat().st_size==count*32
report={'gaussians':count,'ply_bytes':ply.stat().st_size,'splat_bytes':count*32,
        'ply_sha256':hashlib.sha256(ply.read_bytes()).hexdigest(),
        'validation':'Binary PLY length, finite values, positive scales, nonzero quaternions, 32-byte splat record layout checked.',
        'compact_format_note':'SPLAT uses only base colour, losing the PLY spherical-harmonic view-dependent colour terms.'}
(out/'model_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
