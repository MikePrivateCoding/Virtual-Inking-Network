# Patch-level RGB prediction map demo

This package demonstrates region-level inference of VIN for two representative regions.

The TIF files are included only for visual display. Model inference is performed on the corresponding `.npy` feature matrices, which were generated using HIPT (https://github.com/mahmoodlab/HIPT), a hierarchical self-supervised vision transformer for whole-slide pathology images.

For each region, the classifier is applied to 256 HIPT patch features arranged as a 16 × 16 grid. The softmax probability of the cauterized class is used as the patch-level score. Patches with `prob_red >= 0.50` are displayed in red, and patches with `prob_red < 0.50` are displayed in blue.

## Download

Download and extract the [demo package](https://drive.google.com/file/d/1Jx8BRRdizHVDahSBhMx1P8zl7XFpWi4h/view?usp=drivesdk). Copy its `inputs_4096_npy/` and `model/` folders into this repository alongside `4096_region_rgb_demo.py`. Example TIF images are also available in the package.

## Run

Dependencies: Python 3, NumPy, Pillow, and PyTorch.

```bash
python 4096_region_rgb_demo.py
```

## Outputs

The script writes RGB-only prediction maps to `demo_output/`.

## Reference

Chen RJ, Chen C, Li Y, Chen TY, Trister AD, Krishnan RG, Mahmood F. Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning. CVPR 2022.
