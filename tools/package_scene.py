"""Build a standalone local viewer from a compact splat and matching cameras."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--splat', type=Path, required=True)
parser.add_argument('--cameras', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--title', default='Reconstructed room')
args = parser.parse_args()
assert not args.output.exists(), 'Choose a new output directory; overwriting is disabled.'
data = args.splat.read_bytes()
assert data and len(data) % 32 == 0, 'Expected compact 32-byte SPLAT records.'
cameras = json.loads(args.cameras.read_text())
poses = np.asarray(cameras['world_to_camera'], dtype=float)
assert poses.ndim == 3 and poses.shape[1:] == (4, 4) and len(poses) >= 3
assert np.isfinite(poses).all()
centers = np.linalg.inv(poses)[:, :3, 3]
up = -poses[:, :3, :3][:, 1, :].mean(0)
assert np.linalg.norm(up) > 1e-8
up /= np.linalg.norm(up)
scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max())
assert scale > 0
views = []
for label, fraction in [('Overview', .35), ('View 2', .65), ('View 3', .85)]:
    i = round(fraction * (len(poses) - 1))
    views.append({'label': label, 'position': centers[i].tolist(),
                  'target': (centers[i] + poses[i, 2, :3]).tolist()})
fy = float(cameras['intrinsics'][0][1][1])
assert fy > 0 and cameras['height'] > 0
source = Path(__file__).resolve().parents[1] / 'docs'
args.output.mkdir(parents=True)
for name in ['app.js', 'index.html', 'style.css', 'layout.css', 'movement-input.js', 'credits.txt']:
    shutil.copy2(source / name, args.output / name)
shutil.copytree(source / 'vendor', args.output / 'vendor')
# Keep one scene in the standalone preview. Public deployment remains untouched.
index = (args.output / 'index.html').read_text(encoding='utf-8')
start = index.index('<nav class="scene-switcher"')
end = index.index('</nav>', start) + len('</nav>')
index = index[:start] + index[end:]
(args.output / 'index.html').write_text(index, encoding='utf-8')
app = (args.output / 'app.js').read_text(encoding='utf-8')
import re
app = re.sub(r"const scenePath=.*?;", "const scenePath='./scene.json';", app, count=1)
(args.output / 'app.js').write_text(app, encoding='utf-8')
assets = args.output / 'assets'; assets.mkdir()
parts = []
for i, offset in enumerate(range(0, len(data), 8 * 1024 * 1024)):
    path = f'assets/room-{i}.bin'; parts.append(path)
    (args.output / path).write_bytes(data[offset:offset + 8 * 1024 * 1024])
scene = {'name': args.title, 'title': args.title, 'captureLabel': 'Reconstruction',
         'note': 'Local preview. Inspect camera framing and reconstruction quality.',
         'bytes': len(data), 'parts': parts, 'gaussians': len(data) // 32,
         'up': up.tolist(), 'views': views, 'movementScale': scale / 5.66,
         'fov': math.degrees(2 * math.atan(cameras['height'] / (2 * fy))),
         'sha256': hashlib.sha256(data).hexdigest()}
(args.output / 'scene.json').write_text(json.dumps(scene, indent=2), encoding='utf-8')
print(f'Packaged {scene["gaussians"]:,} Gaussians into {args.output}. Serve this directory over HTTP.')
