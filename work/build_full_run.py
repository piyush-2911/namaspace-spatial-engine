"""Build a separate full-video notebook from the verified pilot stages."""
import ast,json
from pathlib import Path
r=Path(__file__).resolve().parent
sfm=(r/'gpu_test.py').read_text()
def replace_once(text,old,new):
    assert old in text, old
    return text.replace(old,new)
for old,new in [
('/kaggle/working/gpu_test','/kaggle/working/full_reconstruction'),
('First 180 seconds, one frame/second, width 1280; initial camera registration test','Full video, one frame/second, width 1600; camera registration'),
("'-t','180','-vf','fps=1,scale=1280:-2'","'-vf','fps=1,scale=1600:-2'"),
("str(images/'frame_%05d.jpg')], 600)","str(images/'frame_%05d.jpg')], 1200)"),
("report['frame_count'] >= 175","report['frame_count'] >= 360"),
('sift.max_num_features=4096','sift.max_num_features=8192'),
('seq.overlap=10','seq.overlap=20'),
("best.export_PLY(r/'sparse_points.ply')","best_dir=r/'sparse_best'\nbest_dir.mkdir(exist_ok=True)\nbest.write(best_dir)\nstats['largest_model_registration_fraction']=best.num_reg_images()/len(list(im.glob('*.jpg')))\nsave()\nassert stats['largest_model_registration_fraction']>=0.9, 'Less than 90% registered: inspect before training'\nbest.export_PLY(r/'sparse_points.ply')"),
("run('sfm', [sys.executable,'-u',str(root/'sfm_worker.py')], 1800)","run('sfm', [sys.executable,'-u',str(root/'sfm_worker.py')], 3600)")]: sfm=replace_once(sfm,old,new)
training=(r/'train_splat.py').read_text()
for old,new in [
("sources=list(Path('/kaggle/input').rglob('sfm_report.json'))","sources=list(Path('/kaggle/working/full_reconstruction').glob('sfm_report.json'))"),
("source/'sparse/0'","source/'sparse_best'"),
('assert rec.num_reg_images()==180',"assert rec.num_reg_images()>=324"),
('(640,round(h*640/w))','(1280,round(h*1280/w))'),
('refine_start_iter=300,refine_stop_iter=2500,reset_every=1500','refine_start_iter=500,refine_stop_iter=6500,reset_every=3000'),
('range(180)','range(len(images))'),
('steps=4000','steps=10000'),
('all 180 images','all registered images'),
("if len(params['means'])>400000","if len(params['means'])>800000"),
('in [1000,4000]','in [1000,4000,10000]'),
("j%4==0","j%8==0")]: training=replace_once(training,old,new)
training+='''
# Preview along recovered capture positions; not a collision-aware walkthrough.
import imageio.v2 as imageio
with imageio.get_writer(str(out/'capture_path_preview.mp4'),fps=15,codec='libx264',quality=7) as writer:
    with torch.no_grad():
        for i in range(0,len(images),2):
            rgb=render(i,3)[0][0].clamp(0,1).cpu().numpy()
            writer.append_data((rgb*255).astype(np.uint8))
print('CAPTURE_PATH_PREVIEW_EXPORTED',flush=True)
'''
setup="""import os,sys,subprocess,json
from pathlib import Path
os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
os.environ['MAX_JOBS']='2'
os.environ['TORCH_EXTENSIONS_DIR']='/kaggle/working/torch_extensions'
"""
train_launch="""
os.environ['CUDA_VISIBLE_DEVICES']='0'
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','gsplat==1.5.3','pytorch-msssim==1.0.0','ninja','imageio[ffmpeg]'],check=True)
p=Path('/kaggle/working/train_full_splat.py');p.write_text(WORKER)
with open('/kaggle/working/training.log','w') as log:
    result=subprocess.run([sys.executable,'-u',str(p)],stdout=log,stderr=subprocess.STDOUT,timeout=9000)
print(Path('/kaggle/working/training.log').read_text()[-16000:])
assert result.returncode==0,'Training failed; inspect training.log'
"""
d=r/'namas-full';d.mkdir(exist_ok=True)
cells=[]
for text in [setup,sfm,'WORKER = '+repr(training)+'\n'+train_launch]:
    ast.parse(text)
    cells.append({'cell_type':'code','metadata':{},'outputs':[],'execution_count':None,'source':text})
ast.parse(training)
# Compile the nested reconstruction worker as well.
tree=ast.parse(sfm)
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='worker' for t in node.targets): ast.parse(node.value.value)
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','metadata':{},'source':'# Namas Full Quality\nFull Meetingroom video → 1600px COLMAP reconstruction → 1280px, 10,000-step Gaussian Splat training. Preserves the previous pilot. Stops if the largest camera reconstruction contains fewer than 90% of extracted frames. Saves PLY, checkpoint, held-out metrics, and a capture-path preview. Collisions and unconstrained free-roaming are not provided by this notebook.'}]+cells}
(d/'namas-full.ipynb').write_text(json.dumps(nb,indent=2))
(d/'sfm_full.py').write_text(sfm)
(d/'train_full.py').write_text(training)
meta={'id':'YOUR_KAGGLE_USERNAME/namas-full-quality','title':'Namas Full Quality','code_file':'namas-full.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,'dataset_sources':['YOUR_KAGGLE_USERNAME/namaspace2'],'kernel_sources':[],'competition_sources':[]}
(d/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
print('Full notebook, reconstruction worker, and training worker syntax validated.')
