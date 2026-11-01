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
credits='NamaSpace prototype\nDataset: Tanks and Temples - Meetingroom.\nCreators: Arno Knapitsch, Jaesik Park, Qian-Yi Zhou and Vladlen Koltun (2017).\nSource: https://www.tanksandtemples.org/download/\nLicense: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated in the Copyright section of https://www.tanksandtemples.org/license/\nLicense link: https://creativecommons.org/licenses/by/4.0/\nChanges by NamaSpace: extracted and resized video frames; estimated camera poses and sparse geometry with COLMAP; trained a Gaussian Splatting reconstruction; exported a compact base-colour model and split it into browser assets.\nThe dataset authors do not endorse or sponsor this project. Source data and derived benchmark assets retain the applicable attribution and license notices; no additional restrictions are imposed by NamaSpace.\nThe source license page also contains conflicting restrictions; this attribution relies on its express CC BY 4.0 statement and does not resolve that inconsistency.\nLiving-room scene: project owner smartphone capture, not Tanks and Temples data.\nTraining: gsplat 1.5.3 (Apache-2.0), https://github.com/nerfstudio-project/gsplat\nThe compact viewer model retains base colour only.\n\n'
for package in ['three','@mkkellogg/gaussian-splats-3d']:
    folder=root/'node_modules'/package
    licenses=list(folder.glob('LICENSE*'))
    credits+=f'\n{package}\n'+''.join(p.read_text(encoding='utf-8') for p in licenses)
(dist/'credits.txt').write_text(credits,encoding='utf-8')
assert b''.join((dist/p).read_bytes() for p in parts)==data
assert all(np.isfinite(v).all() for v in poses)
print(json.dumps({'parts':len(parts),'model_bytes':len(data),'fov':fov,'views':len(views)}))
