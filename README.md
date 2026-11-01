# NamaSpace

Indoor video reconstruction and a browser-based Gaussian splat viewer for the Namma Space competition. Includes the Tanks and Temples Meetingroom benchmark and an approved smartphone living-room reconstruction. The meeting room is the default scene.

## Run the viewer locally

Python 3 is sufficient to serve the included static viewer:

```sh
python -m http.server 8766 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8766/ and wait for the approximately 26 MB scene to load. Click inside the room, drag to look, and use WASD or arrows to move. Q/E changes height, Shift boosts speed, Escape stops input, and R resets the view. Buttons provide saved viewpoints, fullscreen, and movement speed. Touch movement controls appear on narrow screens.

Public viewer: https://piyush-2911.github.io/namaspace-spatial-engine/

GitHub Pages serves `docs/` from `main`. This directory contains the viewer and approved compact models; raw footage and credentials are excluded. Update `docs/` when publishing viewer changes.

## Reconstruct the model and inspect sample data

Follow [RECONSTRUCTION.md](RECONSTRUCTION.md) for the Kaggle setup, input requirements, output files and local model packaging. Real benchmark input frames are included in [test/](test/README.md). The submitted smartphone notebook is [work/namas-living/livingroom.ipynb](work/namas-living/livingroom.ipynb). Raw home recordings are not public.

## Repository contents

- `viewer/dist/`: static viewer, vendored renderer modules and their license notices, compact sample scene, camera presets.
- `viewer/prepare_assets.py`: generates viewer assets from exported reconstruction files; requires NumPy and `npm ci` in `viewer/` to restore the renderer packages.
- `viewer/test-movement.mjs`: regression checks for held keys, focus changes, and touch cancellation. Run `node viewer/test-movement.mjs`.
- `work/namas-full/namas-full.ipynb`: full sample video reconstruction and training notebook for Kaggle with a T4 GPU and internet enabled.
- `work/namas-full/sfm_full.py`, `train_full.py`: corresponding pipeline sources.
- `work/build_*.py`: notebook generation scripts; earlier pilot notebooks are retained for reference.
- `work/prepare_model.py`: converts trained PLY output to the compact base-colour splat format.
- `work/full-results/`: camera poses and small evaluation reports.

## Current results and limitations

The sample reconstruction registered 370 frames. Training used 10,000 steps at 1280×720 on one T4, producing 805,616 Gaussians. Mean evaluation PSNR was 23.97 dB and SSIM 0.859. Evaluation views were held out of photometric training but used in COLMAP camera/seed estimation.

The viewer supports free flight, not collision-aware walking. Chair geometry and fine details remain blurry from some angles. Compact assets omit higher-order view-dependent colour. POI search, path planning and AR navigation are not implemented yet.

## Attribution

Sample: [Tanks and Temples](https://www.tanksandtemples.org/download/) (non-commercial dataset terms apply). Rendering: [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) and [Three.js](https://github.com/mrdoob/three.js). Training: [gsplat](https://github.com/nerfstudio-project/gsplat). See `viewer/dist/credits.txt` for bundled dependency licenses. This repository does not grant additional rights to the dataset or dependencies.

Raw videos, full PLYs, checkpoints, local credentials, and hosting metadata are excluded. Compact benchmark and approved living-room assets are included for public viewing.

## Smartphone reconstruction

Capture 3 registered 198 of 220 frames in the main reconstruction (90%). Training completed 20,000 steps and produced 770,025 Gaussians (24.6 MB compact model). Some blur and incomplete surfaces remain, especially near bright windows.
