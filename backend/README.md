# 3D UNet Inference Service (FastAPI)

This service accepts MRI modalities (T1, T1ce, T2, FLAIR) and performs masked, training-faithful preprocessing and patch-wise stitched inference. It returns segmentation only. Grad-CAM and uncertainty/confidence outputs have been removed for a segmentation-only workflow. It supports both ONNX and PyTorch runtimes.

## Endpoints
- POST /infer:
	- Preferred: `npz_base64`: base64-encoded npz containing arrays with keys `T1`, `T1ce`, `T2`, `FLAIR`, optional `mask`, optional `spacing` (z,y,x).
	- Or small cases: `volumes`: dict of 3D arrays [D,H,W] per modality, optional `mask`, optional `spacing`.
	- Optional: `target_spacing` for resampling.
	- Returns base64 npz payload `seg_npz` (int16 labels 0=background,1=edema,2=necrotic_core,3=enhancing) and `meta` with label dictionary, unique labels, and volume shape.
- GET /health: Model health.

## Quickstart
1. Install dependencies in a Python 3.10+ environment:

```bash
pip install -r backend/requirements.txt
```

2. Models (relative to repo root):
	- ONNX path: `assets/model/segmentation_model.onnx` or env `MODEL_PATH`/`ONNX_MODEL_PATH`.
	- PyTorch path: `assets/model/model.pth` or env `TORCH_MODEL_PATH`/`MODEL_PTH_PATH`.
	The server prefers PyTorch (for Grad-CAM). It falls back to ONNX if PyTorch load fails or torch is unavailable.

3. Run the API server from repo root:

```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

4. Test health:

```bash
curl http://localhost:8000/health
```

## Input contract
- Supported modalities: T1, T1ce, T2, FLAIR (fixed channel order). Missing modalities are zero-filled.
- All volumes must share the same shape [D,H,W].
- Optional mask and spacing (z,y,x). If `target_spacing` provided, the server resamples images (linear) and mask (nearest) before normalization.
- Preprocessing: builds or accepts a brain mask, morphological close/open, robust percentile clipping inside mask, masked z-score per modality, sets out-of-mask voxels to 0.

## Output contract
- `seg_npz`: base64 npz containing `seg` int16 [D,H,W] with labels 0..3.
- `meta`: labels mapping, unique labels found, volume shape, channel order.

## Notes
- Sliding-window: 64x64x64 with 16-voxel overlaps, cosine/Hann blend window for seamless stitching.
- Torch runtime pre-warms once and uses AMP on CUDA when available.
- Background values and NaNs are clamped/cleaned before inference.
