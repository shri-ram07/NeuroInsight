# Brain Tumor Segmentation & Explainability Suite

A full-stack workflow for interactive brain tumor segmentation, visualization, and explainability. The backend serves a 3D UNet inference service with ROI-focused Grad-CAM and voxel-level uncertainty estimation. The Vite/React frontend lets clinicians upload four MRI modalities, inspect segmentations, and blend explainability overlays in real time.

## Key Features

- **Multi-modal 3D UNet inference** with masked preprocessing, patch-wise stitching, and PyTorch/ONNX runtime fallback.
- **ROI-targeted Grad-CAM** that focuses on the highest-confidence tumor region to accelerate and stabilize explainability visualization.
- **Voxel-wise uncertainty maps** (confidence, uncertainty, entropy) to highlight low-certainty regions in the predicted segmentation.
- **Interactive medical viewer** with modality switching, slice navigation, overlay opacity controls, and compact legends that stay out of the primary anatomy view.
- **Hybrid sync/async API**: synchronous explainability path (`/infer`) and queued asynchronous segmentation (`/infer_async`) for large studies.
- **Composable pipelines** for dataset download (`dataPipeLine/`) and modeling experiments (`modelPipeLine/`).

## Architecture Overview

- **Backend (`backend/app.py`)**: FastAPI service wrapping a 3D UNet. Provides preprocessing, sliding-window inference, Grad-CAM ROI generation, and uncertainty computation. Supports both torch and ONNX runtimes.
- **Frontend (`src/`)**: TypeScript + React (Vite) UI. Handles NIfTI uploads, orchestrates backend calls, and renders volumes via canvas overlays in `MedicalViewer`.
- **Assets (`assets/model/`)**: Holds the serialized segmentation models (`segmentation_model.onnx`, `model.pth`).
- **Pipelines**:
  - `dataPipeLine/`: Scripts for fetching and staging MRI volumes.
  - `modelPipeLine/`: Notebooks and helpers for training or evaluating models.
- **Research notes**: `researchPlan/`, `Project_Workflow.md`, and other docs capture experimentation history and planning.

## Prerequisites

- **Python 3.10+** with build tools for `onnxruntime`, `torch`, and SciPy.
- **Node.js 18+** (or Bun) for the frontend.
- GPU is recommended but optional; the service falls back to CPU automatically.
- MRI volumes for all four modalities: T1, T1ce, T2, FLAIR (aligned to the same grid).

## Backend Setup

> The commands below assume Windows PowerShell. Adjust activation commands if you use another shell.

1. **Create and activate a virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. **Install dependencies**:
   ```powershell
   pip install -r backend\requirements.txt
   ```
3. **Place inference models** in `assets\model\`:
   - PyTorch: `model.pth` (preferred for Grad-CAM)
   - ONNX: `segmentation_model.onnx` (fallback / CPU)
   You can override locations via environment variables (`TORCH_MODEL_PATH`, `ONNX_MODEL_PATH`, etc.).
4. **Launch the API** from the repo root:
   ```powershell
   uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
5. **Verify health**:
   ```powershell
   curl http://localhost:8000/health
   ```

## Frontend Setup

1. **Install dependencies** (from repo root):
   ```powershell
   npm install
   ```
   Bun users can run `bun install` instead.
2. **Start the dev server**:
   ```powershell
   npm run dev
   ```
   The app defaults to `http://localhost:5173`. Set `VITE_API_BASE` in a `.env` file if the backend runs on a different host or port.

## Using the Application

1. **Upload volumes**: Drag and drop or browse for NIfTI (`.nii`, `.nii.gz`) files for T1, T1ce, T2, and FLAIR. All four modalities must share the same dimensions.
2. **Choose a workflow**:
   - **Segmentation only**: Leave explainability toggles off. The UI will call `/infer_async` and poll `/progress/{job_id}` until results are ready.
   - **Explainability mode**: Enable the Grad-CAM/uncertainty switch before running. The UI will call `/infer` with `enable_gradcam=true` and render all overlays once the synchronous response arrives.
3. **Inspect results** in `MedicalViewer`:
   - Toggle between input modalities and segmentation output.
   - Adjust opacity sliders for segmentation, Grad-CAM, and uncertainty overlays.
   - Use slice navigation to step through axial slices.
   - Legends stay in the bottom-left corner to avoid obscuring the brain.
4. **Review stats** in the status bar and sidebar, including timing metrics, detected labels, and confidence summaries.

## API Summary

### `POST /infer`
Synchronous inference with explainability. Payload can include either `npz_base64` (preferred) or `volumes` (nested lists). Set `enable_gradcam=true` to receive `gradcam_npz` and `aux_npz` maps. Returns:

- `seg_npz`: base64 NPZ with `seg` volume (labels 0-3)
- `gradcam_npz`: optional NPZ with `gradcam_focus`
- `aux_npz`: NPZ containing `confidence_map`, `uncertainty_map`, `entropy_map`
- `meta`: metadata (label map, shapes, ROI details, timing, uncertainty stats)

### `POST /infer_async`
Queues a background job for segmentation-only inference. Returns `job_id`.

### `GET /progress/{job_id}`
Provides job status and progress percentage.

### `GET /result/{job_id}`
Returns the same payload structure as `/infer` (without Grad-CAM/uncertainty) once the job is complete.

### `GET /health`
Simple health probe indicating whether the PyTorch or ONNX provider is active.

## Directory Reference

- `src/pages/Index.tsx`: Upload workflow, async polling, explainability orchestration.
- `src/components/viewer/MedicalViewer.tsx`: Canvas renderer for modalities, segmentations, Grad-CAM, and uncertainty overlays.
- `src/components/layout/ControlSidebar.tsx`: Toggles for overlays, explainability mode, and opacity sliders.
- `src/lib/npz.ts`: Helpers to decode base64-compressed NPZ payloads in the browser.
- `backend/app.py`: FastAPI service, preprocessing, sliding-window inference, ROI Grad-CAM generation, uncertainty math, async job queue.
- `dataPipeLine/`: Download utilities for curated MRI datasets.
- `modelPipeLine/`: Training notebooks and scripts (`main_PipeLine.ipynb`, `model.py`).
- `assets/images/`: Static media for documentation or UI.

## Explainability Details

- **ROI selection**: The backend identifies the voxel with the highest combined tumor probability and confidence, crops a fixed ROI, and runs Grad-CAM focused on that region to shorten runtime.
- **Uncertainty**: Converts softmax logits to probability volumes, then derives confidence (maximum probability), uncertainty (`1 - confidence`), and entropy. The frontend normalizes each slice to preserve contrast.
- **Overlay rendering**: Grad-CAM uses a warm colormap, while uncertainty uses a teal-to-magenta ramp blended in screen mode. Opacity defaults balance visibility with anatomical detail and can be tuned per user.

## Troubleshooting

- **Grad-CAM unavailable**: Ensure `model.pth` is accessible. The service returns a 400 error if only ONNX is available for explainability requests.
- **Shape mismatches**: All four modalities must match exactly. Resample externally or supply `target_spacing` in the request so the backend can resample.
- **Slow explainability**: ROI-based Grad-CAM reduces runtime, but large volumes still take longer. Monitor `timings_ms` in the response metadata.
- **Frontend build errors**: Delete `node_modules`, clear the Vite cache (`npm run clean` if defined), reinstall dependencies.

## Roadmap Ideas

- Batch explainability for multiple ROIs or user-selected voxels.
- Persisted studies with precomputed overlays.
- Extended uncertainty metrics (e.g., mutual information for MC Dropout).
- Automated QA checks within `dataPipeLine/` for dataset validation.

---

For further implementation notes, see `Project_Workflow.md` and the inline comments within the core viewer components.
