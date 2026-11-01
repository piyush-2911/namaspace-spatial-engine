# Reconstruct a scene

## 1. Check the submitted viewer and sample data

From the repository root, with Python 3 installed:

```sh
python -m http.server 8766 --bind 127.0.0.1 --directory docs
```

Open http://127.0.0.1:8766/. `docs/` includes both approved models. The older `viewer/dist/` directory contains the benchmark model only.

The `test/meetingroom/images/` directory contains 24 real input frames from the public Tanks and Temples Meetingroom video. These are a small input-format/feature-matching sample, not the complete training set. See `test/README.md` for attribution, timestamps, checksums and validation.

## 2. Prepare the reconstruction input

Use the full video for an end-to-end reproduction:

| Run | Notebook | Expected input filename | Settings |
| --- | --- | --- | --- |
| Meetingroom benchmark | `work/namas-full/namas-full.ipynb` | `Meetingroom.mp4` | 1 fps, 1600px SfM images; 10,000 training steps at 1280px |
| Submitted smartphone pipeline | `work/namas-living/livingroom.ipynb` | `livingroom3.mp4` | 2 fps, 1600px SfM images; 20,000 training steps at 1440px |

Download the full benchmark video from https://www.tanksandtemples.org/download/ and follow its license. The original home recording is intentionally not included publicly. To run the smartphone pipeline, supply your own indoor video named `livingroom3.mp4` in a **private Kaggle dataset**. Different footage will produce different results.

The benchmark notebook expects at least 360 extracted frames and checks decoding around 360 seconds. The smartphone notebook expects at least 200 frames and checks decoding around 100 seconds. These are capture-specific safeguards, not universal defaults. For footage of a different duration, edit the input filename, tail-decode timestamp, minimum frame count and minimum registered-image count in the notebook. Both notebooks stop if the largest reconstruction registers less than 90% of extracted frames; inspect coverage and camera matching before changing that gate.

Capture overlapping views while moving slowly through the room. Keep the camera lens/zoom fixed, avoid rapid turns and moving people, and explicitly cover ceilings and other surfaces you want reconstructed. Unobserved surfaces cannot be recovered reliably.

## 3. Run on Kaggle

1. Create a notebook and import the appropriate `.ipynb` above.
2. Add your video dataset through **Add Input**. Do not attach both input videos to a run.
3. Enable **Internet** and select **GPU T4 x2**. The pipeline uses GPU 0; it does not distribute training across both GPUs. The code checks that a T4 is available.
4. Run all cells in order, or use **Save Version / Run All**. Run in a fresh session to avoid mixing old COLMAP databases with a different capture.
5. Check the reconstruction report before interpreting training results. A failed registration gate means training did not complete successfully.

The notebook installs `pycolmap-cuda12==4.2.1`, `gsplat==1.5.3`, `pytorch-msssim==1.0.0`, Ninja and imageio/FFmpeg support. It uses the Kaggle environment's PyTorch, CUDA, OpenCV, NumPy, SciPy and Pillow. The CUDA extension may take time to compile on the first run. A local CPU-only environment is not a substitute for this training environment.

Kaggle's web interface needs no credentials in the notebook. Optional CLI metadata uses `YOUR_KAGGLE_USERNAME` placeholders; replace those only in your local copy. Never commit `kaggle.json`, tokens, notebook outputs containing account details, or raw private recordings.

## 4. Inspect and download the outputs

Outputs are saved beneath `/kaggle/working/`:

- `full_reconstruction/sfm_report.json`: registration counts, separate components and reconstruction diagnostics.
- `full_reconstruction/sparse_best/` and `images/`: recovered camera/point model and extracted images.
- `splat_output/checkpoint.pt`: model and optimizer state.
- `splat_output/meetingroom.splat` or `livingroom.splat`: compact browser model.
- `splat_output/cameras.json`: matching camera poses and intrinsics.
- `splat_output/metrics.json`, `comparison_*.jpg`, `capture_path_preview.mp4`: evaluation and visual review.
- `training.log`: training progress and failures.

Download the `.splat` and `cameras.json` from the **same run**. Do not mix model and camera coordinates from different reconstructions. Evaluation frames are held out of photometric optimization, but COLMAP uses all registered views to estimate poses and seed geometry.

## 5. Build a local viewer for the new output

Install NumPy locally (`python -m pip install numpy`), then run:

```sh
python tools/package_scene.py --splat path/to/livingroom.splat --cameras path/to/cameras.json --output preview-dist --title "Reconstructed room"
python -m http.server 8767 --bind 127.0.0.1 --directory preview-dist
```

Open http://127.0.0.1:8767/. The packager copies the renderer, chunks the model and derives saved viewpoints from the supplied cameras. It creates a standalone single-scene preview and refuses to overwrite an existing output directory. Inspect and adjust viewpoints: an arbitrary capture position may face a wall.

This command does not publish anything. The deployed site serves `docs/` on `main`; only replace its model assets after reviewing the new reconstruction.

## Verification and limits

The submitted 20k smartphone run completed with 198/220 registered frames and 770,025 Gaussians. A separate 30k continuation brought only a small metric improvement and is not the deployed model. These historical results are not a guarantee for new footage.

The repository's sample checks verify file integrity and decoding. They do not constitute a fresh GPU reconstruction test. The viewer supports free flight, not collision avoidance. POI search, optimal path planning and AR remain future work.
