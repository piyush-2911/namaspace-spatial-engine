import ast,json
from pathlib import Path
r=Path(__file__).resolve().parent
code=(r/'gpu_test.py').read_text()
ast.parse(code)
nb=json.loads((r/'namas/namas.original.ipynb').read_text())
nb['cells'] += [{'cell_type':'markdown','metadata':{},'source':'## T4 camera reconstruction pilot\nChecks the accelerator and video ending; extracts the first 180 seconds at 1 fps; runs CUDA SIFT extraction, sequential matching, and sparse reconstruction. This is a subset pilot, not the full room or a Gaussian splat.'},{'cell_type':'code','metadata':{},'source':code,'outputs':[],'execution_count':None}]
nb['metadata']['kaggle']['isGpuEnabled']=True
nb['metadata']['kaggle']['accelerator']='gpu'
(r/'namas/namas.ipynb').write_text(json.dumps(nb,indent=2))
meta=json.loads((r/'namas/kernel-metadata.json').read_text())
meta['enable_gpu']=True
(r/'namas/kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('GPU pilot syntax checked and notebook prepared.')
