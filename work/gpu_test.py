import json, subprocess, sys, time, shutil, traceback
from pathlib import Path

root = Path('/kaggle/working/gpu_test')
root.mkdir(exist_ok=True)
report = {'stages': {}, 'scope': 'First 180 seconds, one frame/second, width 1280; initial camera registration test'}
def save():
    (root/'report.json').write_text(json.dumps(report, indent=2))
def run(name, args, timeout):
    started = time.time()
    with (root/f'{name}.log').open('w') as log:
        result = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    report['stages'][name] = {'returncode': result.returncode, 'seconds': round(time.time()-started, 2)}
    save()
    print(name, report['stages'][name], flush=True)
    if result.returncode:
        print((root/f'{name}.log').read_text()[-6000:])
        raise RuntimeError(f'{name} failed: see saved log')
try:
    run('gpu', ['nvidia-smi'], 30)
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'], text=True)
    report['gpu'] = gpu
    print(gpu, flush=True)
    assert 'T4' in gpu, 'Expected T4; stopping to avoid using an unintended accelerator'
    video = next(Path('/kaggle/input').rglob('Meetingroom.mp4'))
    # A separate tail check distinguishes OpenCV seeking issues from FFmpeg decode failures.
    run('tail_decode', ['ffmpeg','-hide_banner','-v','warning','-ss','360','-i',str(video),'-t','10','-f','null','-'], 180)
    images=root/'images'
    images.mkdir(exist_ok=True)
    run('extract', ['ffmpeg','-hide_banner','-v','warning','-i',str(video),'-t','180','-vf','fps=1,scale=1280:-2','-q:v','2',str(images/'frame_%05d.jpg')], 600)
    report['frame_count'] = len(list(images.glob('*.jpg')))
    assert report['frame_count'] >= 175, 'Unexpected frame count'
    save()
    run('install', [sys.executable,'-m','pip','install','--disable-pip-version-check','pycolmap-cuda12==4.2.1'], 300)
    worker = r'''
import pycolmap as pc, json, time
from pathlib import Path
r=Path('/kaggle/working/gpu_test')
stats={'version':pc.__version__,'has_cuda':pc.has_cuda}
def save(): (r/'sfm_report.json').write_text(json.dumps(stats,indent=2))
save()
assert pc.has_cuda, 'PyCOLMAP wheel has no CUDA support'
print('PyCOLMAP', stats, flush=True)
db=r/'database.db'
im=r/'images'
ext=pc.FeatureExtractionOptions() if hasattr(pc,'FeatureExtractionOptions') else pc.SiftExtractionOptions()
sift=ext.sift if hasattr(ext,'sift') else ext
sift.max_num_features=4096
if hasattr(ext,'num_threads'): ext.num_threads=4
t=time.time()
kwargs={'extraction_options':ext} if 'extraction_options' in pc.extract_features.__doc__ else {'sift_options':ext}
reader=pc.ImageReaderOptions()
reader.camera_model='SIMPLE_RADIAL'
pc.extract_features(db, im, camera_mode=pc.CameraMode.SINGLE, reader_options=reader, device=pc.Device.cuda, **kwargs)
stats['extraction_seconds']=time.time()-t
save()
match=pc.FeatureMatchingOptions() if hasattr(pc,'FeatureMatchingOptions') else pc.SiftMatchingOptions()
if hasattr(match,'num_threads'): match.num_threads=4
seq=pc.SequentialPairingOptions() if hasattr(pc,'SequentialPairingOptions') else pc.SequentialMatchingOptions()
seq.overlap=10
seq.loop_detection=False
kwargs={'matching_options':match} if 'matching_options' in pc.match_sequential.__doc__ else {'sift_options':match}
t=time.time()
pc.match_sequential(db, pairing_options=seq, device=pc.Device.cuda, **kwargs)
stats['matching_seconds']=time.time()-t
save()
output=r/'sparse'
output.mkdir(exist_ok=True)
t=time.time()
maps=pc.incremental_mapping(db, im, output)
stats['mapping_seconds']=time.time()-t
stats['models']=[{'id':int(k),'registered_images':v.num_reg_images(),'points3D':v.num_points3D(),'mean_reprojection_error':v.compute_mean_reprojection_error()} for k,v in maps.items()]
save()
print(json.dumps(stats,indent=2),flush=True)
assert maps, 'No reconstruction recovered'
best=max(maps.values(),key=lambda v:v.num_reg_images())
best.export_PLY(r/'sparse_points.ply')
'''
    (root/'sfm_worker.py').write_text(worker)
    run('sfm', [sys.executable,'-u',str(root/'sfm_worker.py')], 1800)
    report['result']='completed'
except Exception as e:
    report['result']='failed'
    report['error']=str(e)
    traceback.print_exc()
    raise
finally:
    save()
