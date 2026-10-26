import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'namas'
path = root / 'namas.ipynb'
original = path.read_text(encoding='utf-8')
(root / 'namas.original.ipynb').write_text(original, encoding='utf-8')
nb = json.loads(original)
code = r'''
import json, sys, shutil, subprocess, importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

out = Path('/kaggle/working/diagnostics')
out.mkdir(exist_ok=True)
videos = sorted(p for p in Path('/kaggle/input').rglob('*') if p.suffix.lower() in {'.mp4','.mov','.mkv','.avi','.m4v'})
report = {'python': sys.version, 'tools': {x: shutil.which(x) for x in ['ffmpeg','ffprobe','colmap','nvidia-smi']},
          'packages': {x: importlib.util.find_spec(x) is not None for x in ['torch','nerfstudio','gsplat','pycolmap']}, 'videos': []}
if report['packages']['torch']:
    import torch
    report['torch'] = {'version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_version': torch.version.cuda}
assert videos, 'No supported video files in the attached dataset'
for vi, video in enumerate(videos):
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened(), f'Cannot open {video.name}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    item = {'path': str(video), 'bytes': video.stat().st_size, 'fps': fps, 'frame_count': count,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration_seconds': count / fps if fps else None, 'samples': []}
    assert count > 0 and fps > 0, 'Invalid video metadata'
    sheet = Image.new('RGB', (4 * 320, 3 * 205), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, frame_index in enumerate(np.linspace(0, count-2, 12).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = cap.read()
        if not ok:
            item['samples'].append({'frame': int(frame_index), 'decoded': False})
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (640, max(1, round(gray.shape[0] * 640 / gray.shape[1]))))
        sample = {'frame': int(frame_index), 'seconds': round(frame_index/fps, 2), 'decoded': True,
                  'laplacian_variance_640w': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                  'mean_brightness': float(gray.mean())}
        item['samples'].append(sample)
        im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        im.thumbnail((320,180))
        x,y=(i%4)*320,(i//4)*205
        sheet.paste(im,(x,y))
        draw.text((x+4,y+182), f'{sample["seconds"]:.1f}s | sharpness {sample["laplacian_variance_640w"]:.0f}', fill='black')
    cap.release()
    sheet.save(out / f'video_{vi}_contact_sheet.jpg')
    item['sample_decode_pass'] = all(x['decoded'] for x in item['samples'])
    report['videos'].append(item)
    print(json.dumps(item, indent=2))
    display(sheet)
(out / 'report.json').write_text(json.dumps(report, indent=2))
print('DIAGNOSTICS_COMPLETE', json.dumps({'tools': report['tools'], 'packages': report['packages']}))
assert all(v['sample_decode_pass'] for v in report['videos']), 'Some sample frames could not be decoded; inspect report'
'''
nb['cells'] += [
    {'cell_type':'markdown','metadata':{},'source':'## Meeting-room input diagnostics\nCPU-only checks: video metadata, 12 distributed decode samples, visual coverage contact sheet, and installed reconstruction tools. Sharpness values are descriptive, not a pass/fail guarantee. This does not yet validate camera registration or 3D reconstruction.'},
    {'cell_type':'code','metadata':{},'source':code,'outputs':[],'execution_count':None}
]
path.write_text(json.dumps(nb, indent=2), encoding='utf-8')
compile(code, '<diagnostic-cell>', 'exec')
print('Diagnostic cell syntax checked; original notebook backed up.')
