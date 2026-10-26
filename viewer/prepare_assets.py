from pathlib import Path
import json,shutil,hashlib
import numpy as np
root=Path(__file__).resolve().parent
dist=root/'dist';vendor=dist/'vendor';vendor.mkdir(exist_ok=True)
shutil.copy2(root/'node_modules/three/build/three.module.js',vendor/'three.module.js')
shutil.copy2(root/'node_modules/@mkkellogg/gaussian-splats-3d/build/gaussian-splats-3d.module.js',vendor/'gaussian-splats-3d.module.js')
data=(root.parent/'outputs/full-quality/meetingroom.splat').read_bytes()
parts=[]
for i,start in enumerate(range(0,len(data),8*1024*1024)):
    name=f'assets/room-{i}.bin';(dist/name).write_bytes(data[start:start+8*1024*1024]);parts.append(name)
cameras=json.loads((root.parent/'work/full-results/cameras.json').read_text())
poses=np.array(cameras['world_to_camera'])
up=-poses[:,:3,:3][:,1,:].mean(0);up/=np.linalg.norm(up)
views=[]
for label,index in [('Main room',64),('Window side',128),('Across the tables',16),('Ceiling detail',256)]:
    pose=poses[index];inv=np.linalg.inv(pose);position=inv[:3,3];forward=pose[2,:3]
    views.append({'label':label,'position':position.tolist(),'target':(position+forward).tolist()})
fov=float(np.degrees(2*np.arctan(cameras['height']/(2*cameras['intrinsics'][64][1][1]))))
scene={'name':'Meetingroom sample','bytes':len(data),'parts':parts,'gaussians':len(data)//32,'up':up.tolist(),'views':views,'fov':fov,'movementScale':1.0,'sha256':hashlib.sha256(data).hexdigest()}
(dist/'scene.json').write_text(json.dumps(scene,indent=2))
credits='NamaSpace prototype\nDataset: Tanks and Temples Meetingroom, https://www.tanksandtemples.org/download/ (non-commercial use; review dataset license before other uses).\nTraining: gsplat 1.5.3 (Apache-2.0), https://github.com/nerfstudio-project/gsplat\nThe compact viewer model retains base colour only.\n\n'
for package in ['three','@mkkellogg/gaussian-splats-3d']:
    folder=root/'node_modules'/package
    licenses=list(folder.glob('LICENSE*'))
    credits+=f'\n{package}\n'+''.join(p.read_text(encoding='utf-8') for p in licenses)
(dist/'credits.txt').write_text(credits,encoding='utf-8')
assert b''.join((dist/p).read_bytes() for p in parts)==data
assert all(np.isfinite(v).all() for v in poses)
print(json.dumps({'parts':len(parts),'model_bytes':len(data),'fov':fov,'views':len(views)}))
