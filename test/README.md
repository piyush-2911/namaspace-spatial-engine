# Sample input data

`meetingroom/images/` contains **24 JPEG frames** sampled at 1 fps from approximately 60-83 seconds of the public Tanks and Temples **Meetingroom** video, resized to 960 pixels wide. These are real input images, not model renders. No home recording is included.

This small sample is suitable for inspecting the input format and experimenting with feature extraction/matching. It does **not** contain enough coverage to reproduce the full submitted room. Use the full video and the instructions in [RECONSTRUCTION.md](../RECONSTRUCTION.md) for the complete pipeline; do not feed this subset into the full-video notebook's frame-count gates.

## Validate the sample

```sh
python -m pip install Pillow
python test/validate_sample.py
```

The check validates checksums, JPEG decoding, dimensions and metadata. The expected result is 24 valid frames. `meetingroom/manifest.json` records provenance, sample timestamps and SHA-256 checksums.

## Attribution and terms

Source: Tanks and Temples, Meetingroom benchmark scene, by Arno Knapitsch, Jaesik Park, Qian-Yi Zhou and Vladlen Koltun (2017), https://www.tanksandtemples.org/download/.

Declared license: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), per the source Copyright section. No endorsement by the dataset authors is implied. NamaSpace imposes no additional restrictions on these benchmark frames.

License and use terms: https://www.tanksandtemples.org/license/. The source page identifies CC BY 4.0 and also contains non-commercial research restrictions; preserve the original terms and consult the dataset authors for uses outside this educational demonstration. We do not grant additional rights to the data.

Changes: selected a 24-second interval, extracted JPEG frames at 1 fps, resized to 960px width, and removed embedded metadata. This is public benchmark data, **not** the team's own capture.
