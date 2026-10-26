# NamaSpace Room Explorer

Static browser viewer. Automatically loads the full-video sample's compact Gaussian Splat model. Free-fly movement, touch controls, saved interior viewpoints, reset, speed, fullscreen, loading progress, error recovery, and renderer FPS.

## Run locally

Serve `dist/` over HTTP, for example `python -m http.server 8765 --bind 127.0.0.1 --directory dist`. Open `http://127.0.0.1:8765`.

## Replace the scene

The app reads `dist/scene.json`. Replace its binary assets and update `parts`, `bytes`, camera `views`, normalized `up`, and `fov`. Coordinates must be from the same reconstruction as the new model. Do not reuse sample camera positions for a different room. `prepare_assets.py` demonstrates assembly using the saved COLMAP camera export and compact `.splat` asset.

Rendering is client-side; no Kaggle session or API credential is used by this website. The model is split into small static files and reassembled in browser memory. No user upload is required to open the sample.

The sample is labeled as public benchmark data. This prototype has free-flight navigation, not collision-aware walking. It uses a base-colour model without higher-order spherical harmonics. Dataset and dependency attribution is in `dist/credits.txt`. Test on judge devices before submission; large splat scenes can be demanding on mobile GPUs.

## Local viewer quality fix
Camera matrices are updated before the initial splat sort and viewpoint jumps. During free flight, sorting is refreshed at most every 100 ms after movement or look input, rather than relying on the library's larger movement thresholds. CPU worker sorting is used for browser compatibility. A full-precision PLY comparison did not resolve the initial blur; correcting the stale camera transform did. Some chair distortion remains in the training render itself. No new training was run for this fix.

Keep this project local until the user explicitly approves deployment. The active test URL is http://127.0.0.1:8766/.


Movement keys are scoped to the focused room: click the scene before WASD. Escape, key release, focus changes, tab hiding and pointer cancellation clear held input. Keyboard and touch holds are tracked separately. Run node test-movement.mjs from viewer to check release and focus regressions.

