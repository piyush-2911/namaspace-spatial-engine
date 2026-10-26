import ast,json
from pathlib import Path
r=Path(__file__).resolve().parent
worker=(r/'train_splat.py').read_text(); ast.parse(worker)
setup='''import os, sys, subprocess, json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
os.environ['MAX_JOBS']='2'
os.environ['TORCH_EXTENSIONS_DIR']='/kaggle/working/torch_extensions'
subprocess.run(['nvidia-smi'],check=True)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','gsplat==1.5.3','pycolmap==4.2.1','pytorch-msssim==1.0.0','ninja'],check=True)
'''
launcher='''p=Path('/kaggle/working/train_splat.py')
p.write_text(WORKER)
with open('/kaggle/working/training.log','w') as log:
    result=subprocess.run([sys.executable,'-u',str(p)],stdout=log,stderr=subprocess.STDOUT,timeout=5400)
print(Path('/kaggle/working/training.log').read_text()[-12000:])
assert result.returncode==0, 'Training failed; inspect training.log'
'''
d=r/'namas-train'; d.mkdir(exist_ok=True)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[
{'cell_type':'markdown','metadata':{},'source':'# Meeting-room Gaussian Splat pilot\nReads the successful namas reconstruction. Standard gsplat 1.5.3 rasterization and DefaultStrategy; custom minimal training loop with L1 + SSIM loss. 4,000 iterations at 640px; every eighth view held out of photometric training. Exports PLY, checkpoints, comparisons, and metrics. One T4 is used. Source: https://github.com/nerfstudio-project/gsplat/tree/v1.5.3'},
{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':setup},
{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':'WORKER = '+repr(worker)+'\n'+launcher}]}
(d/'namas-train.ipynb').write_text(json.dumps(nb,indent=2))
meta={'id':'YOUR_KAGGLE_USERNAME/namas-train','title':'Namas Gaussian Splat Training','code_file':'namas-train.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'dataset_sources':[],'kernel_sources':['YOUR_KAGGLE_USERNAME/namas'],'competition_sources':[]}
(d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
for cell in nb['cells']:
    if cell['cell_type']=='code': ast.parse(cell['source'])
print('Notebook and embedded training code passed syntax checks.')
