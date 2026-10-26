"""Small reproducible gsplat baseline. Uses gsplat's standard densification strategy.
Reference: https://github.com/nerfstudio-project/gsplat/tree/v1.5.3
Input: saved COLMAP pilot, not a new capture. Coordinates remain in COLMAP units.
"""
import os, json, time, random
from pathlib import Path
import cv2, numpy as np, torch, pycolmap
from PIL import Image
from scipy.spatial import cKDTree
from pytorch_msssim import ssim
from gsplat import rasterization, export_splats
from gsplat.strategy import DefaultStrategy

torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.set_num_threads(4)
out=Path('/kaggle/working/splat_output'); out.mkdir(exist_ok=True)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'T4' in torch.cuda.get_device_name(0), 'Expected T4'
sources=list(Path('/kaggle/working/full_reconstruction').glob('sfm_report.json'))
assert len(sources)==1, f'Expected one reconstruction, got {sources}'
source=sources[0].parent
rec=pycolmap.Reconstruction(source/'sparse_best')
assert rec.num_reg_images()>=324
images=sorted(rec.images.values(),key=lambda i:i.name)
frames=[]; views=[]; intrinsics=[]; centers=[]; names=[]
for im in images:
    cam=rec.cameras[im.camera_id]
    assert cam.model.name=='SIMPLE_RADIAL', cam.model.name
    rgb=cv2.imread(str(source/'images'/im.name))
    assert rgb is not None
    rgb=cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)
    h,w=rgb.shape[:2]
    rgb=cv2.resize(rgb,(1280,round(h*1280/w)),interpolation=cv2.INTER_AREA)
    h,w=rgb.shape[:2]
    K=cam.calibration_matrix().copy(); K[0]*=w/cam.width; K[1]*=h/cam.height
    dist=np.array([cam.params[3],0.,0.,0.])
    newK,_=cv2.getOptimalNewCameraMatrix(K,dist,(w,h),0,(w,h))
    rgb=cv2.undistort(rgb,K,dist,None,newK)
    view=np.eye(4,dtype=np.float32); view[:3]=im.cam_from_world().matrix()
    centers.append(np.linalg.inv(view)[:3,3])
    frames.append(torch.from_numpy(rgb.copy()))
    views.append(view); intrinsics.append(newK); names.append(im.name)
views=torch.tensor(np.stack(views),device='cuda',dtype=torch.float32)
intrinsics=torch.tensor(np.stack(intrinsics),device='cuda',dtype=torch.float32)
centers=np.stack(centers)
scene_scale=float(np.max(np.linalg.norm(centers-centers.mean(0),axis=1)))
points=list(rec.points3D.values())
xyz=np.array([p.xyz for p in points],dtype=np.float32)
colors=np.array([p.color for p in points],dtype=np.float32)/255
distances=cKDTree(xyz).query(xyz,k=4)[0][:,1:]
scales=np.log(np.maximum(np.sqrt((distances**2).mean(1)),1e-6))
def param(x): return torch.nn.Parameter(torch.as_tensor(x,dtype=torch.float32,device='cuda'))
n=len(points)
params=torch.nn.ParameterDict({
    'means':param(xyz), 'scales':param(np.repeat(scales[:,None],3,axis=1)),
    'quats':param(np.random.rand(n,4)), 'opacities':param(np.full(n,np.log(.1/.9))),
    'sh0':param(((colors-.5)/.28209479177387814)[:,None]), 'shN':param(np.zeros((n,15,3)))})
lrs={'means':1.6e-4*scene_scale,'scales':.005,'quats':.001,'opacities':.05,'sh0':.0025,'shN':.0025/20}
optim={k:torch.optim.Adam([v],lr=lrs[k],eps=1e-15) for k,v in params.items()}
strategy=DefaultStrategy(refine_start_iter=500,refine_stop_iter=6500,reset_every=3000,refine_every=100,verbose=False)
strategy.check_sanity(params,optim)
state=strategy.initialize_state(scene_scale=scene_scale)
train=[i for i in range(len(images)) if i%8!=0]; test=[i for i in range(len(images)) if i%8==0]
steps=10000
config={'steps':steps,'width':w,'height':h,'train_images':len(train),'held_out_images':len(test),
        'seed':42,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'source':str(source),'scene_scale':scene_scale,'gsplat':'1.5.3',
        'evaluation_note':'Views held out of photometric training; COLMAP poses and seed geometry used all registered images.'}
(out/'config.json').write_text(json.dumps(config,indent=2))
(out/'cameras.json').write_text(json.dumps({'names':names,'world_to_camera':views.cpu().tolist(),'intrinsics':intrinsics.cpu().tolist(),'width':w,'height':h}))
print('DATA_READY',json.dumps(config),flush=True)
def render(i,degree):
    return rasterization(means=params['means'],quats=params['quats'],scales=params['scales'].exp(),
        opacities=params['opacities'].sigmoid(),colors=torch.cat([params['sh0'],params['shN']],1),
        viewmats=views[i:i+1],Ks=intrinsics[i:i+1],width=w,height=h,sh_degree=degree,
        near_plane=.01,far_plane=1e6,packed=False)
def export(step):
    data={k:v.detach().cpu() for k,v in params.items()}
    assert all(torch.isfinite(v).all() for v in data.values())
    torch.save({'step':step,'splats':data,'config':config,'optimizers':{k:v.state_dict() for k,v in optim.items()},'strategy_state':state},out/'checkpoint.pt')
    export_splats(**data,format='ply',save_to=str(out/'meetingroom.ply'))
    print('EXPORTED',step,len(data['means']),flush=True)
started=time.time()
for step in range(steps):
    if step%len(train)==0: random.shuffle(train)
    i=train[step%len(train)]
    gt=frames[i].to('cuda',dtype=torch.float32)[None]/255
    degree=min(step//1000,3)
    rendered,alpha,info=render(i,degree)
    strategy.step_pre_backward(params,optim,state,step,info)
    loss=.8*(rendered-gt).abs().mean()+.2*(1-ssim(rendered.permute(0,3,1,2),gt.permute(0,3,1,2),data_range=1,size_average=True))
    assert torch.isfinite(loss), 'Non-finite training loss'
    loss.backward()
    for opt in optim.values(): opt.step(); opt.zero_grad(set_to_none=True)
    optim['means'].param_groups[0]['lr']=lrs['means']*(.01**((step+1)/steps))
    if len(params['means'])>800000: strategy.refine_stop_iter=step
    strategy.step_post_backward(params,optim,state,step,info,packed=False)
    if step%100==0:
        status={'step':step,'loss':float(loss.detach()),'splats':len(params['means']),'elapsed_seconds':time.time()-started}
        (out/'progress.json').write_text(json.dumps(status,indent=2)); print(status,flush=True)
    if step+1 in [1000,4000,10000]: export(step+1)
metrics=[]
with torch.no_grad():
    for j,i in enumerate(test):
        pred=render(i,3)[0].clamp(0,1)
        gt=frames[i].to('cuda',dtype=torch.float32)[None]/255
        mse=(pred-gt).square().mean()
        score={'image':names[i],'psnr':float(-10*torch.log10(mse)),
               'ssim':float(ssim(pred.permute(0,3,1,2),gt.permute(0,3,1,2),data_range=1,size_average=True))}
        metrics.append(score)
        if j%8==0:
            pair=torch.cat([gt[0],pred[0]],dim=1).cpu().numpy()
            Image.fromarray((pair*255).astype(np.uint8)).save(out/f'comparison_{j:02d}.jpg')
summary={'result':'completed','steps':steps,'gaussians':len(params['means']),
         'training_and_evaluation_seconds':time.time()-started,'psnr_mean':float(np.mean([m['psnr'] for m in metrics])),
         'ssim_mean':float(np.mean([m['ssim'] for m in metrics])),'views':metrics,
         'ply_bytes':(out/'meetingroom.ply').stat().st_size,'max_gpu_allocated_gb':torch.cuda.max_memory_allocated()/1024**3}
(out/'metrics.json').write_text(json.dumps(summary,indent=2))
print('TRAINING_COMPLETE',json.dumps(summary),flush=True)

# Preview along recovered capture positions; not a collision-aware walkthrough.
import imageio.v2 as imageio
with imageio.get_writer(str(out/'capture_path_preview.mp4'),fps=15,codec='libx264',quality=7) as writer:
    with torch.no_grad():
        for i in range(0,len(images),2):
            rgb=render(i,3)[0][0].clamp(0,1).cpu().numpy()
            writer.append_data((rgb*255).astype(np.uint8))
print('CAPTURE_PATH_PREVIEW_EXPORTED',flush=True)
