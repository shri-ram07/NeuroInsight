import io
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import threading
import uuid
import base64

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import onnxruntime as ort
from scipy.ndimage import binary_closing, binary_opening
from scipy.ndimage import zoom as ndi_zoom
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

app = FastAPI(title="3D UNet Inference Service", version="0.1.0")

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- Models & Schemas ---------

class InferenceRequest(BaseModel):
    # Option 1 (preferred): single base64 npz containing arrays: T1, T1ce, T2, FLAIR, optional mask, optional spacing
    npz_base64: Optional[str] = Field(default=None, description="Base64-encoded npz with keys T1,T1ce,T2,FLAIR and optional mask,spacing")
    # Option 2: raw nested lists per modality (small cases)
    volumes: Optional[Dict[str, List[List[List[float]]]]] = None
    mask: Optional[List[List[List[int]]]] = None
    spacing: Optional[Tuple[float, float, float]] = None
    # If provided, hint for canonical spacing to resample to
    target_spacing: Optional[Tuple[float, float, float]] = None
    # Enable explainability mode (Grad-CAM)
    enable_gradcam: Optional[bool] = Field(default=False, description="Enable Grad-CAM explainability analysis")

class InferenceResponse(BaseModel):
    # Base64-encoded npz payload for compact transfer
    seg_npz: str
    # Grad-CAM heatmap (if enabled)
    gradcam_npz: Optional[str] = None
    # Metadata for client rendering (use Any for flexible structure)
    meta: Dict[str, Any]

# --------- Runtime Session ---------

class OnnxRuntime:
    def __init__(self, onnx_path: str, providers: Optional[List[str]] = None):
        opts = ort.SessionOptions()
        opts.enable_mem_pattern = False
        if not os.path.exists(onnx_path):
            raise RuntimeError(f"ONNX model not found at '{onnx_path}'. Place your model at this path or update backend/app.py.")
        prov = providers or [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        try:
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=prov)
        except Exception:
            self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]) 
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, patch: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: patch})
        return outputs[0]


if TORCH_AVAILABLE:
    class firstConv3d(nn.Module):
        def __init__(self, input_size, kernel_size=3):
            super().__init__()
            self.conv1 = nn.Conv3d(input_size, 32, kernel_size, padding=1)
            self.bn1 = nn.BatchNorm3d(32)
            self.conv2 = nn.Conv3d(32, 32, kernel_size, padding=1)
            self.bn2 = nn.BatchNorm3d(32)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x

    class downSampling(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            skip = x
            x = self.pool(x)
            return x, skip

    class bottleNeck(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_ch)
            self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_ch)
            self.dropout = nn.Dropout3d(p=0.3)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout(x)
            return x

    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv3d(F_g, F_int, kernel_size=1),
                nn.BatchNorm3d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1),
                nn.BatchNorm3d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1),
                nn.BatchNorm3d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class upSampling(nn.Module):
        def __init__(self, in_channels, skip_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.attention = AttentionGate(F_g=out_channels, F_l=skip_channels, F_int=out_channels // 2)
            self.conv1 = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(out_channels)
            self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            if x.shape[2:] != skip.shape[2:]:
                diffZ = skip.size(2) - x.size(2)
                diffY = skip.size(3) - x.size(3)
                diffX = skip.size(4) - x.size(4)
                x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2,
                              diffZ // 2, diffZ - diffZ // 2])
            skip = self.attention(x, skip)
            x = torch.cat([skip, x], dim=1)
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            return x

    class uNET(nn.Module):
        def __init__(self, in_channel, out_channel):
            super().__init__()
            self.dovConv = firstConv3d(in_channel)
            self.d1 = downSampling(32, 64)
            self.d2 = downSampling(64, 128)
            self.d3 = downSampling(128, 256)
            self.d4 = downSampling(256, 512)
            self.b1 = bottleNeck(512, 512)
            self.u1 = upSampling(512, 512, 256)
            self.u2 = upSampling(256, 256, 128)
            self.u3 = upSampling(128, 128, 64)
            self.u4 = upSampling(64, 64, 32)
            self.f = nn.Conv3d(32, out_channel, kernel_size=1)

        def forward(self, x):
            x = self.dovConv(x)
            x1, skip1 = self.d1(x)
            x2, skip2 = self.d2(x1)
            x3, skip3 = self.d3(x2)
            x4, skip4 = self.d4(x3)
            x5 = self.b1(x4)
            x = self.u1(x5, skip4)
            x = self.u2(x, skip3)
            x = self.u3(x, skip2)
            x = self.u4(x, skip1)
            out = self.f(x)
            return out

    class TorchRuntime:
        def __init__(self, pth_path: str, device: Optional[str] = None):
            if not os.path.exists(pth_path):
                raise RuntimeError(f"PyTorch model not found at '{pth_path}'.")
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
            # Try load full model first; if that fails, instantiate and load state_dict
            load_errs = []
            model: Optional[nn.Module] = None
            try:
                obj = torch.load(pth_path, map_location=self.device)
                if isinstance(obj, nn.Module):
                    model = obj
                elif isinstance(obj, dict):
                    # Could be raw state_dict or a wrapper dict
                    state_dict = None
                    if all(isinstance(k, str) for k in obj.keys()) and any(k.startswith('dovConv') or k.startswith('f') for k in obj.keys()):
                        state_dict = obj  # looks like state_dict
                    elif 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                        state_dict = obj['state_dict']
                    else:
                        # Last resort: try using the dict as state_dict
                        state_dict = obj
                    model = uNET(in_channel=4, out_channel=4)
                    model.load_state_dict(state_dict, strict=False)
                else:
                    load_errs.append(f"Unsupported .pth object type: {type(obj)}")
            except Exception as e:
                load_errs.append(str(e))
            if model is None:
                raise RuntimeError("Failed to load PyTorch model: " + "; ".join(load_errs))
            self.model: nn.Module = model.to(self.device)
            self.model.eval()
            # Initialize Medical Grad-CAM for explainability with multiple decoder layers
            self.gradcam = MedicalGradCAM(self.model, target_layers=['u3.conv2', 'u4.conv2'])
            # Pre-warm with a dummy patch to stabilize kernels
            with torch.no_grad():
                dummy = torch.zeros((1, 4, 64, 64, 64), dtype=torch.float32, device=self.device)
                _ = self.model(dummy)

        def infer(self, patch: np.ndarray, amp: bool = True, enable_gradcam: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
            """
            Unified inference method that can optionally compute Grad-CAM
            
            Args:
                patch: Input patch [1,C,D,H,W] float32 numpy
                amp: Whether to use automatic mixed precision
                enable_gradcam: Whether to compute Grad-CAM heatmaps
                
            Returns:
                If enable_gradcam=False: segmentation_output (np.ndarray)
                If enable_gradcam=True: (segmentation_output, gradcam_heatmaps) (Tuple)
            """
            # Convert to tensor
            tensor = torch.from_numpy(patch)
            if self.device.type == 'cuda':
                tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            else:
                tensor = tensor.to(self.device)
            
            if enable_gradcam:
                # Enable gradients for Grad-CAM computation
                tensor.requires_grad_(True)
                
                # Forward pass (no torch.no_grad() when gradcam is needed)
                if self.device.type == 'cuda' and amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(tensor)
                else:
                    output = self.model(tensor)
                
                # Get segmentation result
                segmentation = output.detach().float().cpu().numpy()
                
                # Generate medical Grad-CAM heatmaps using the same forward pass
                roi_heatmaps = self.gradcam.generate_medical_roi_heatmaps(
                    tensor, target_classes=[1, 2, 3], model_output=output
                )
                
                return segmentation, roi_heatmaps
            else:
                # Standard inference without gradients (more efficient)
                with torch.no_grad():
                    if self.device.type == 'cuda' and amp:
                        with torch.cuda.amp.autocast():
                            output = self.model(tensor)
                    else:
                        output = self.model(tensor)
                    
                    return output.detach().float().cpu().numpy()

class MedicalGradCAM:
    """
    Medical Grad-CAM implementation specifically designed for 3D brain tumor segmentation
    Uses multiple target layers and region-focused scoring for better explainability
    """
    def __init__(self, model: torch.nn.Module, target_layers: List[str] = None):
        self.model = model
        # Use multiple layers from the decoder path for better localization
        if target_layers is None:
            target_layers = ['u3.conv2', 'u4.conv2']  # Use actual decoder conv layers
        self.target_layers = target_layers
        self.layer_gradients = {}
        self.layer_activations = {}
        self.hooks = []
        
        # Register hooks for multiple layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks for multiple target layers"""
        available_layers = [name for name, _ in self.model.named_modules()]
        print(f"Available model layers: {available_layers}")
        print(f"Looking for target layers: {self.target_layers}")
        
        for layer_name in self.target_layers:
            layer_found = False
            # Create closures to capture the layer name
            def make_forward_hook(name):
                def forward_hook(module, input, output):
                    print(f"Forward hook triggered for {name}, output shape: {output.shape}")
                    self.layer_activations[name] = output.detach()
                return forward_hook
            
            def make_backward_hook(name):
                def backward_hook(module, grad_input, grad_output):
                    if grad_output[0] is not None:
                        print(f"Backward hook triggered for {name}, grad shape: {grad_output[0].shape}")
                        self.layer_gradients[name] = grad_output[0].detach()
                return backward_hook
            
            # Find and hook the target layers
            for name, module in self.model.named_modules():
                if name == layer_name:
                    self.hooks.append(module.register_forward_hook(make_forward_hook(layer_name)))
                    self.hooks.append(module.register_backward_hook(make_backward_hook(layer_name)))
                    print(f"Successfully registered hooks for layer: {layer_name}")
                    layer_found = True
                    break
            
            if not layer_found:
                print(f"WARNING: Layer '{layer_name}' not found in model!")
    
    def generate_cam_for_class(self, input_tensor: torch.Tensor, class_idx: int, segmentation_mask: torch.Tensor = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class with medical segmentation focus
        
        Args:
            input_tensor: Input tensor of shape [1, C, D, H, W]
            class_idx: Target class index (1: necrotic, 2: edema, 4: enhancing)
            segmentation_mask: Optional segmentation mask to focus on tumor regions
        
        Returns:
            cam: Grad-CAM heatmap as numpy array of shape [D, H, W]
        """
        # Enable gradients for input
        input_tensor.requires_grad_(True)
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)  # [1, num_classes, D, H, W]
        
        # Clear previous gradients
        self.model.zero_grad()
        
        # Create target score focused on tumor regions
        if segmentation_mask is not None:
            # Focus on regions where the class is predicted
            class_output = output[:, class_idx, :, :, :]  # [1, D, H, W]
            # Use segmentation mask to focus on relevant regions
            tumor_mask = (segmentation_mask == class_idx).float()
            print(f"Class {class_idx}: tumor_mask sum = {tumor_mask.sum().item()}, class_output range = [{class_output.min().item():.4f}, {class_output.max().item():.4f}]")
            if tumor_mask.sum() > 0:
                # Weighted average focusing on predicted tumor regions
                target_score = (class_output * tumor_mask).sum() / (tumor_mask.sum() + 1e-8)
            else:
                # Fallback to max activation if no tumor regions
                target_score = torch.max(class_output)
        else:
            # Use spatial-weighted approach for better localization
            class_output = output[:, class_idx, :, :, :]
            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=1)[:, class_idx, :, :, :]
            # Weighted sum focusing on high-confidence regions
            target_score = torch.sum(probs * class_output)
        
        print(f"Class {class_idx}: target_score = {target_score.item():.6f}")
        
        # Backward pass
        target_score.backward(retain_graph=True)
        
        # Check if gradients were captured
        print(f"Class {class_idx}: Captured gradients for layers: {list(self.layer_gradients.keys())}")
        print(f"Class {class_idx}: Captured activations for layers: {list(self.layer_activations.keys())}")
        
        # Generate CAM from multiple layers and combine
        combined_cam = None
        layer_count = 0
        
        for layer_name in self.target_layers:
            if layer_name in self.layer_gradients and layer_name in self.layer_activations:
                gradients = self.layer_gradients[layer_name]  # [1, C, D', H', W']
                activations = self.layer_activations[layer_name]  # [1, C, D', H', W']
                
                print(f"  Layer {layer_name}: gradients shape={gradients.shape}, grad_range=[{gradients.min().item():.6f}, {gradients.max().item():.6f}]")
                print(f"  Layer {layer_name}: activations shape={activations.shape}, act_range=[{activations.min().item():.6f}, {activations.max().item():.6f}]")
                
                # Compute importance weights for each feature map
                weights = torch.mean(gradients, dim=[2, 3, 4], keepdim=True)  # [1, C, 1, 1, 1]
                
                # Generate CAM for this layer
                layer_cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, D', H', W']
                layer_cam = torch.nn.functional.relu(layer_cam)
                
                print(f"  Layer {layer_name}: layer_cam range=[{layer_cam.min().item():.6f}, {layer_cam.max().item():.6f}]")
                
                # Resize to match input dimensions
                if layer_cam.shape[2:] != input_tensor.shape[2:]:
                    layer_cam = torch.nn.functional.interpolate(
                        layer_cam, 
                        size=input_tensor.shape[2:], 
                        mode='trilinear', 
                        align_corners=False
                    )
                
                # Combine CAMs from multiple layers
                if combined_cam is None:
                    combined_cam = layer_cam
                else:
                    combined_cam += layer_cam
                layer_count += 1
        
        if combined_cam is not None and layer_count > 0:
            # Average across layers
            combined_cam = combined_cam / layer_count
            
            # Apply ReLU and normalize
            combined_cam = torch.nn.functional.relu(combined_cam)
            combined_cam = combined_cam.squeeze()  # [D, H, W]
            
            print(f"  Final CAM for class {class_idx}: shape={combined_cam.shape}, range=[{combined_cam.min().item():.6f}, {combined_cam.max().item():.6f}]")
            
            # Normalize to [0, 1] and apply scaling for better visibility
            if combined_cam.max() > 0:
                combined_cam = (combined_cam - combined_cam.min()) / (combined_cam.max() - combined_cam.min())
                print(f"  Normalized CAM for class {class_idx}: range=[{combined_cam.min().item():.6f}, {combined_cam.max().item():.6f}]")
                
                # Apply exponential scaling to enhance visibility of small values
                # This helps make subtle patterns more visible in the frontend
                combined_cam = torch.pow(combined_cam, 0.5)  # Square root scaling for better visibility
                print(f"  Enhanced CAM for class {class_idx}: range=[{combined_cam.min().item():.6f}, {combined_cam.max().item():.6f}]")
            else:
                print(f"  WARNING: CAM for class {class_idx} has zero max value!")
            
            return combined_cam.cpu().numpy()
        
        # Return zeros if no valid CAM was generated
        return np.zeros(input_tensor.shape[2:])  # [D, H, W]
    
    def generate_medical_roi_heatmaps(self, input_tensor: torch.Tensor, target_classes: List[int] = None, model_output: torch.Tensor = None) -> Dict[str, np.ndarray]:
        """
        Generate ROI heatmaps for multiple brain tumor classes using medical-focused approach
        
        Args:
            input_tensor: Input tensor of shape [1, C, D, H, W]
            target_classes: List of class indices to generate heatmaps for
            model_output: Optional pre-computed model output to avoid redundant forward pass
        
        Returns:
            roi_heatmaps: Dictionary with class names as keys and heatmaps as values
        """
        if target_classes is None:
            # Use 4-class model indices: 0=Background, 1=NCR/NET, 2=Edema, 3=Enhancing Tumor
            target_classes = [1, 2, 3]
        
        # Map to medical terminology for 4-class model
        class_names = {
            0: 'background',  # Background
            1: 'necrotic',    # NCR/NET (Necrotic and Non-Enhancing Tumor)
            2: 'edema',       # ED (Peritumoral Edema) 
            3: 'enhancing'    # ET (Enhancing Tumor)
        }
        
        roi_heatmaps = {}
        
        # Use provided model output or compute it if not provided
        if model_output is not None:
            initial_output = model_output
            segmentation_pred = torch.argmax(initial_output, dim=1, keepdim=True)  # [1, 1, D, H, W]
        else:
            # Fallback: compute model output (only when not provided)
            with torch.no_grad():
                initial_output = self.model(input_tensor)
                segmentation_pred = torch.argmax(initial_output, dim=1, keepdim=True)  # [1, 1, D, H, W]
        
        for class_idx in target_classes:
            if class_idx in class_names:
                # Generate class-specific heatmap with segmentation guidance
                heatmap = self.generate_cam_for_class(
                    input_tensor, 
                    class_idx, 
                    segmentation_mask=segmentation_pred.squeeze(0).squeeze(0)  # [D, H, W]
                )
                roi_heatmaps[class_names[class_idx]] = heatmap
        
        return roi_heatmaps
    
    def cleanup(self):
        """Remove registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

# Lazy global runtime (path configurable)
_runtime_onnx: Optional[OnnxRuntime] = None
_runtime_torch: Optional['TorchRuntime'] = None
_last_torch_error: Optional[str] = None


def _resolve_model_path() -> str:
    # Prefer environment variables if set
    env_path = os.environ.get("MODEL_PATH") or os.environ.get("ONNX_MODEL_PATH")
    if env_path:
        return env_path
    # Default: repo-root/assets/model/segmentation_model.onnx
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "assets", "model", "segmentation_model.onnx")


def _resolve_pth_path() -> str:
    env_path = os.environ.get("TORCH_MODEL_PATH") or os.environ.get("MODEL_PTH_PATH")
    if env_path:
        return env_path
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(repo_root, "assets", "model", "model.pth")


def get_runtime(prefer: str = "torch") -> Union['TorchRuntime', OnnxRuntime]:
    """Return preferred runtime. Prefer torch for Grad-CAM if available, else ONNX."""
    global _runtime_torch, _runtime_onnx, _last_torch_error
    
    print(f"get_runtime called with prefer='{prefer}', TORCH_AVAILABLE={TORCH_AVAILABLE}")
    
    if prefer == "torch" and TORCH_AVAILABLE:
        if _runtime_torch is None:
            try:
                pth_path = _resolve_pth_path()
                print(f"Trying to load PyTorch model from: {pth_path}")
                _runtime_torch = TorchRuntime(pth_path)
                print("PyTorch runtime initialized successfully!")
                return _runtime_torch
            except Exception as e:
                # Fallback to onnx
                _last_torch_error = str(e)
                print(f"PyTorch runtime failed: {e}")
        else:
            print("Returning existing PyTorch runtime")
            return _runtime_torch
    
    print("Falling back to ONNX runtime")
    if _runtime_onnx is None:
        _runtime_onnx = OnnxRuntime(_resolve_model_path())
    return _runtime_onnx

# --------- Volume Utilities ---------

PATCH = (64, 64, 64)  # D, H, W
STRIDE = (48, 48, 48)  # overlap 16 voxels per axis
BLEND_WINDOW: Optional[np.ndarray] = None  # [1,1,pD,pH,pW] blend weights cached

def zyx_shape(arr: np.ndarray) -> Tuple[int, int, int]:
    assert arr.ndim == 3, f"Expected 3D, got {arr.shape}"
    return int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])


def _build_mask(vols: Dict[str, np.ndarray], provided_mask: Optional[np.ndarray]) -> np.ndarray:
    if provided_mask is not None:
        mask = (provided_mask > 0).astype(np.uint8)
    else:
        # Default: union of non-zero voxels across modalities
        keys = [k for k in ["T1", "T1ce", "T2", "FLAIR"] if k in vols]
        mask = np.zeros_like(vols[keys[0]], dtype=np.uint8)
        for k in keys:
            mask |= (vols[k] != 0).astype(np.uint8)
    # Morphological clean-up: close then open to fill holes and remove speckles
    try:
        mask = binary_closing(mask, iterations=1)
        mask = binary_opening(mask, iterations=1)
    except Exception:
        pass
    return mask.astype(np.uint8)


def _masked_zscore(v: np.ndarray, mask: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    v = v.astype(np.float32)
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(v, dtype=np.float32)
    vox = v[m]
    lo, hi = np.percentile(vox, [p_low, p_high])
    v_clip = np.clip(v, lo, hi)
    mu = float(v_clip[m].mean())
    sd = float(v_clip[m].std())
    sd = sd if sd > 0 else 1.0
    v_z = (v_clip - mu) / sd
    v_z[~m] = 0.0
    # Clean NaNs
    v_z = np.nan_to_num(v_z, nan=0.0, posinf=0.0, neginf=0.0)
    return v_z


def stack_modalities(volumes: Dict[str, np.ndarray], mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    # Fixed modality order: T1, T1ce, T2, FLAIR (missing -> zeros)
    order = ["T1", "T1ce", "T2", "FLAIR"]
    present = [volumes[k] for k in order if k in volumes]
    if not present:
        raise HTTPException(status_code=400, detail="No volumes found in request")
    dz, dy, dx = zyx_shape(present[0])
    # Validate shapes
    for k, v in volumes.items():
        if v.shape != (dz, dy, dx):
            raise HTTPException(status_code=400, detail=f"Volume {k} has shape {v.shape}, expected {(dz, dy, dx)}")
    # Build or use provided mask
    m = _build_mask(volumes, mask)
    channels: List[np.ndarray] = []
    for k in order:
        if k in volumes:
            channels.append(_masked_zscore(volumes[k], m))
        else:
            channels.append(np.zeros((dz, dy, dx), dtype=np.float32))
    stacked = np.stack(channels, axis=0).astype(np.float32)  # [C, D, H, W]
    return stacked, m.astype(np.uint8)


def sliding_window_coords(shape: Tuple[int, int, int], patch: Tuple[int, int, int], stride: Tuple[int, int, int]):
    D, H, W = shape
    pD, pH, pW = patch
    sD, sH, sW = stride
    def gen(limit, size, step):
        if limit <= size:
            yield 0
        else:
            pos = 0
            while pos + size < limit:
                yield pos
                pos += step
            yield max(0, limit - size)
    for z in gen(D, pD, sD):
        for y in gen(H, pH, sH):
            for x in gen(W, pW, sW):
                yield z, y, x


def _get_blend_window() -> np.ndarray:
    global BLEND_WINDOW
    if BLEND_WINDOW is not None:
        return BLEND_WINDOW
    pD, pH, pW = PATCH
    def hann(n):
        if n == 1:
            return np.ones((1,), dtype=np.float32)
        x = np.arange(n, dtype=np.float32)
        return 0.5 * (1 - np.cos(2 * np.pi * x / (n - 1)))
    wz = hann(pD)
    wy = hann(pH)
    wx = hann(pW)
    w3d = wz[:, None, None] * wy[None, :, None] * wx[None, None, :]
    w3d = w3d.astype(np.float32)
    w3d /= (w3d.max() + 1e-8)
    BLEND_WINDOW = w3d[None, None, ...]  # [1,1,pD,pH,pW]
    return BLEND_WINDOW


def stitch_logits(logits_sum: np.ndarray, weights_sum: np.ndarray, logits_patch: np.ndarray, z: int, y: int, x: int):
    # Weighted accumulation to avoid seams
    K, pD, pH, pW = logits_patch.shape
    w = _get_blend_window()[0, 0, :pD, :pH, :pW]
    logits_sum[:, z:z+pD, y:y+pH, x:x+pW] += logits_patch * w[None, ...]
    weights_sum[z:z+pD, y:y+pH, x:x+pW] += w


def _resample_if_needed(vols: Dict[str, np.ndarray], mask: Optional[np.ndarray], spacing: Optional[Tuple[float, float, float]], target_spacing: Optional[Tuple[float, float, float]]):
    """Resample images and mask to target_spacing if provided (linear for images, nearest for mask)."""
    if spacing is None or target_spacing is None:
        return vols, mask
    sz, sy, sx = spacing
    tz, ty, tx = target_spacing
    if sz <= 0 or sy <= 0 or sx <= 0 or tz <= 0 or ty <= 0 or tx <= 0:
        return vols, mask
    # Compute zoom factors per axis (D,H,W align with spacing tuple)
    zoom_factors = (sz / tz, sy / ty, sx / tx)
    out_vols: Dict[str, np.ndarray] = {}
    for k, v in vols.items():
        out_vols[k] = ndi_zoom(v, zoom=zoom_factors, order=1).astype(np.float32)
    out_mask = None
    if mask is not None:
        out_mask = ndi_zoom(mask.astype(np.uint8), zoom=zoom_factors, order=0).astype(np.uint8)
    return out_vols, out_mask


def infer_volume(stacked: np.ndarray, runtime: Union['TorchRuntime', OnnxRuntime]) -> np.ndarray:
    # stacked: [C, D, H, W]
    C, D, H, W = stacked.shape
    pD, pH, pW = PATCH
    # Probe for class channels
    probe = np.zeros((1, C, pD, pH, pW), dtype=np.float32)
    
    # Handle unified method for TorchRuntime
    if isinstance(runtime, TorchRuntime):
        out = runtime.infer(probe, enable_gradcam=False)
    else:
        out = runtime.infer(probe)
        
    if out.ndim != 5:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape {out.shape}, expected [1, K, d, h, w]")
    K = out.shape[1]

    logits_sum = np.zeros((K, D, H, W), dtype=np.float32)
    weights_sum = np.zeros((D, H, W), dtype=np.float32)

    for z, y, x in sliding_window_coords((D, H, W), PATCH, STRIDE):
        patch = stacked[:, z:z+pD, y:y+pH, x:x+pW]
        pad_z = pD - patch.shape[1]
        pad_y = pH - patch.shape[2]
        pad_x = pW - patch.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            patch = np.pad(patch, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='constant')
        patch_b = patch[None, ...].astype(np.float32)
        
        # Use unified method for TorchRuntime, standard method for OnnxRuntime  
        if isinstance(runtime, TorchRuntime):
            logits = runtime.infer(patch_b, enable_gradcam=False)[0]
        else:
            logits = runtime.infer(patch_b)[0]
            
        logits = logits[:, :min(pD, D - z), :min(pH, H - y), :min(pW, W - x)]
        stitch_logits(logits_sum, weights_sum, logits, z, y, x)

    weights_sum = np.clip(weights_sum, 1e-6, None)
    logits_avg = logits_sum / weights_sum[None, ...]

    # Argmax segmentation
    seg = np.argmax(logits_avg, axis=0).astype(np.int16)
    return seg


def infer_volume_with_gradcam(stacked: np.ndarray, runtime: 'TorchRuntime') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Perform volume inference with Grad-CAM heatmap generation for explainability
    
    Args:
        stacked: Input volume [C, D, H, W]
        runtime: TorchRuntime instance with Grad-CAM support
        
    Returns:
        Tuple of (segmentation, gradcam_heatmaps_dict)
    """
    # stacked: [C, D, H, W]
    C, D, H, W = stacked.shape
    pD, pH, pW = PATCH
    
    # Probe for class channels
    probe = np.zeros((1, C, pD, pH, pW), dtype=np.float32)
    out = runtime.infer(probe)
    if out.ndim != 5:
        raise HTTPException(status_code=500, detail=f"Unexpected model output shape {out.shape}, expected [1, K, d, h, w]")
    K = out.shape[1]

    logits_sum = np.zeros((K, D, H, W), dtype=np.float32)
    weights_sum = np.zeros((D, H, W), dtype=np.float32)
    
    # Grad-CAM heatmaps for each class
    gradcam_heatmaps = {
        'necrotic': np.zeros((D, H, W), dtype=np.float32),
        'edema': np.zeros((D, H, W), dtype=np.float32),
        'enhancing': np.zeros((D, H, W), dtype=np.float32)
    }
    gradcam_weights = np.zeros((D, H, W), dtype=np.float32)

    # Process patches with sliding window
    for z, y, x in sliding_window_coords((D, H, W), PATCH, STRIDE):
        patch = stacked[:, z:z+pD, y:y+pH, x:x+pW]
        pad_z = pD - patch.shape[1]
        pad_y = pH - patch.shape[2]
        pad_x = pW - patch.shape[3]
        if pad_z > 0 or pad_y > 0 or pad_x > 0:
            patch = np.pad(patch, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='constant')
        patch_b = patch[None, ...].astype(np.float32)
        
        # Unified inference - compute both segmentation and Grad-CAM in single pass
        # Use Grad-CAM more frequently for better coverage and positioning accuracy
        if z % STRIDE[0] == 0 and y % STRIDE[1] == 0 and x % STRIDE[2] == 0:
            # Single forward pass for both segmentation and Grad-CAM
            try:
                result = runtime.infer(patch_b, enable_gradcam=True)
                logits, roi_heatmaps = result
                logits = logits[0]  # Remove batch dimension
                
                # Stitch segmentation logits
                logits = logits[:, :min(pD, D - z), :min(pH, H - y), :min(pW, W - x)]
                stitch_logits(logits_sum, weights_sum, logits, z, y, x)
                
                # Stitch Medical Grad-CAM heatmaps
                actual_z = min(pD, D - z)
                actual_y = min(pH, H - y)
                actual_x = min(pW, W - x)
                
                for class_name, heatmap in roi_heatmaps.items():
                    if class_name in gradcam_heatmaps:
                        heatmap_patch = heatmap[:actual_z, :actual_y, :actual_x]
                        gradcam_heatmaps[class_name][z:z+actual_z, y:y+actual_y, x:x+actual_x] += heatmap_patch
                        
                gradcam_weights[z:z+actual_z, y:y+actual_y, x:x+actual_x] += 1.0
            except Exception as e:
                print(f"Unified inference failed for patch at ({z},{y},{x}): {e}")
                # Fallback to segmentation-only for this patch
                logits = runtime.infer(patch_b, enable_gradcam=False)[0]
                logits = logits[:, :min(pD, D - z), :min(pH, H - y), :min(pW, W - x)]
                stitch_logits(logits_sum, weights_sum, logits, z, y, x)
        else:
            # Segmentation-only for patches without Grad-CAM (for speed)
            logits = runtime.infer(patch_b, enable_gradcam=False)[0]
            logits = logits[:, :min(pD, D - z), :min(pH, H - y), :min(pW, W - x)]
            stitch_logits(logits_sum, weights_sum, logits, z, y, x)

    # Normalize segmentation
    weights_sum = np.clip(weights_sum, 1e-6, None)
    logits_avg = logits_sum / weights_sum[None, ...]
    seg = np.argmax(logits_avg, axis=0).astype(np.int16)
    
    # Normalize Grad-CAM heatmaps
    gradcam_weights = np.clip(gradcam_weights, 1e-6, None)
    for class_name in gradcam_heatmaps:
        gradcam_heatmaps[class_name] = gradcam_heatmaps[class_name] / gradcam_weights
        # Additional normalization to [0, 1]
        heatmap = gradcam_heatmaps[class_name]
        if heatmap.max() > 0:
            gradcam_heatmaps[class_name] = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return seg, gradcam_heatmaps


# Grad-CAM removed: segmentation-only service

# --------- API Endpoints ---------

@app.post("/infer", response_model=InferenceResponse)
def infer_endpoint(req: InferenceRequest):
    # Load inputs (either npz_base64 or nested lists)
    try:
        np_vols: Dict[str, np.ndarray]
        mask_np: Optional[np.ndarray] = None
        spacing = req.spacing
        if req.npz_base64:
            raw = base64.b64decode(req.npz_base64)
            with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
                keys = list(npz.keys())
                np_vols = {}
                for k in ["T1", "T1ce", "T2", "FLAIR"]:
                    if k in npz:
                        np_vols[k] = np.array(npz[k], dtype=np.float32)
                if 'mask' in npz:
                    mask_np = np.array(npz['mask']).astype(np.uint8)
                if 'spacing' in npz and spacing is None:
                    spacing_arr = np.array(npz['spacing']).astype(np.float32).tolist()
                    if isinstance(spacing_arr, list) and len(spacing_arr) == 3:
                        spacing = (float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2]))
        else:
            if not req.volumes:
                raise HTTPException(status_code=400, detail="Provide either npz_base64 or volumes")
            np_vols = {k: np.asarray(v, dtype=np.float32) for k, v in req.volumes.items()}
            if req.mask is not None:
                mask_np = np.asarray(req.mask, dtype=np.uint8)
        # Validate same shape among present modalities
        shapes = {k: v.shape for k, v in np_vols.items()}
        if len(np_vols) == 0:
            raise HTTPException(status_code=400, detail="No volumes provided")
        if len({tuple(s) for s in shapes.values()}) != 1:
            raise HTTPException(status_code=400, detail=f"All volumes must share same shape, got: {shapes}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid inputs: {e}")

    # Optional resampling if spacing and target spacing provided
    np_vols, mask_np = _resample_if_needed(np_vols, mask_np, spacing, req.target_spacing)

    # Preprocess with masked z-score
    stacked, mask_used = stack_modalities(np_vols, mask_np)  # [C,D,H,W], [D,H,W]

    # Prepare NPZ payload helper
    def _to_npz_b64(data_dict: Dict[str, np.ndarray]) -> str:
        buf = io.BytesIO()
        np.savez_compressed(buf, **data_dict)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    # Check if Grad-CAM is requested
    if req.enable_gradcam:
        # Grad-CAM inference requires PyTorch runtime
        runtime = get_runtime(prefer="torch")
        if not isinstance(runtime, TorchRuntime):
            raise HTTPException(status_code=400, detail="Grad-CAM explainability requires PyTorch runtime")
        
        # Perform inference with Grad-CAM
        seg, gradcam_heatmaps = infer_volume_with_gradcam(stacked, runtime)
        
        # Prepare segmentation NPZ
        seg_npz = _to_npz_b64({'seg': seg.astype(np.int16)})
        
        # Prepare Grad-CAM NPZ with ROI heatmaps (add debugging)
        gradcam_data = {}
        for class_name, heatmap in gradcam_heatmaps.items():
            heatmap_float32 = heatmap.astype(np.float32)
            gradcam_data[f'gradcam_{class_name}'] = heatmap_float32
            print(f"Encoding {class_name} heatmap: shape={heatmap_float32.shape}, "
                  f"range=[{heatmap_float32.min():.6f}, {heatmap_float32.max():.6f}], "
                  f"dtype={heatmap_float32.dtype}, non_zero_count={np.count_nonzero(heatmap_float32)}")
        
        gradcam_npz = _to_npz_b64(gradcam_data)
        
        uniq = sorted([int(x) for x in np.unique(seg)])
        meta = {
            "labels": {"0": "background", "1": "edema", "2": "necrotic_core", "3": "enhancing"},
            "unique_labels": uniq,
            "shape": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
            "output_shapes": {
                "seg": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
                "gradcam_necrotic": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
                "gradcam_edema": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
                "gradcam_enhancing": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
            },
            "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
            "gradcam_classes": ["necrotic", "edema", "enhancing"],
            "explainability_mode": True,
        }
        
        return InferenceResponse(
            seg_npz=seg_npz,
            gradcam_npz=gradcam_npz,
            meta=meta,
        )
    else:
        # Standard segmentation inference
        runtime = get_runtime(prefer="torch")
        seg = infer_volume(stacked, runtime)
        
        # Prepare NPZ payload
        seg_npz = _to_npz_b64({'seg': seg.astype(np.int16)})

        uniq = sorted([int(x) for x in np.unique(seg)])
        meta = {
            "labels": {"0": "background", "1": "edema", "2": "necrotic_core", "3": "enhancing"},
            "unique_labels": uniq,
            "shape": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
            "output_shapes": {
                "seg": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
            },
            "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
            "explainability_mode": False,
        }

        return InferenceResponse(
            seg_npz=seg_npz,
            meta=meta,
        )

@app.get("/health")
def health():
    try:
        rt = get_runtime()
        provider = "torch" if TORCH_AVAILABLE and isinstance(rt, TorchRuntime) else "onnx"
        return {"status": "ok", "provider": provider}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/status")
def status():
    """Return detailed backend status including runtime, model loading, Grad-CAM availability, and config."""
    info: Dict[str, object] = {"status": "ok"}
    # Runtime
    try:
        rt = get_runtime()
        is_torch = TORCH_AVAILABLE and isinstance(rt, TorchRuntime)
        runtime_info = {"type": "torch" if is_torch else "onnx"}
        if is_torch:
            # Torch specifics
            model_path = _resolve_pth_path()
            device = str(rt.device)
            runtime_info.update({
                "model_path": model_path,
                "device": device,
                "warmed_up": True,
            })
        else:
            # ONNX specifics
            onnx_path = _resolve_model_path()
            providers = []
            try:
                providers = rt.session.get_providers()
            except Exception:
                pass
            runtime_info.update({
                "model_path": onnx_path,
                "providers": providers,
                "torch_load_error": _last_torch_error,
            })
        info["runtime"] = runtime_info
    except Exception as e:
        info["runtime_error"] = str(e)

    # Preprocessing config
    info["preprocessing"] = {
        "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
        "mask_strategy": "provided_mask_or_union_nonzero",
        "morphology": "closing_then_opening",
        "normalization": {
            "type": "masked_zscore",
            "clip_percentiles": [1, 99],
            "background_to_zero": True,
        },
        "resampling": {
            "enabled": True,
            "image_interp": "linear",
            "mask_interp": "nearest",
        },
    }

    # Inference config
    info["inference"] = {
        "patch": list(PATCH),
        "stride": list(STRIDE),
        "blend_window": "hann",
        "blend_cached": BLEND_WINDOW is not None,
    }

    # Output schema
    info["outputs"] = {
        "labels": {"0": "background", "1": "edema", "2": "necrotic_core", "3": "enhancing"},
        "payloads": ["seg_npz"],
    }

    # Versions
    info["versions"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "onnxruntime": ort.__version__,
        "torch": torch.__version__ if TORCH_AVAILABLE else None,
        "scipy": None,
    }
    try:
        import scipy
        info["versions"]["scipy"] = scipy.__version__
    except Exception:
        pass
    return info

# --------- Async Inference with Progress ---------

JobsLock = threading.Lock()
Jobs: Dict[str, Dict[str, object]] = {}


def _count_patches(shape: Tuple[int, int, int]) -> int:
    return sum(1 for _ in sliding_window_coords(shape, PATCH, STRIDE))


def _infer_job(job_id: str, np_vols: Dict[str, np.ndarray]):
    try:
        # Debug: print incoming volume shapes
        try:
            print(f"[job {job_id}] starting with modalities: {list(np_vols.keys())}")
            print(f"[job {job_id}] input shapes: {{k: v.shape for k,v in np_vols.items()}}")
        except Exception:
            pass
        stacked, _ = stack_modalities(np_vols)  # [C,D,H,W]
        C, D, H, W = stacked.shape
        try:
            print(f"[job {job_id}] stacked shape: C={C}, D={D}, H={H}, W={W}")
        except Exception:
            pass
        total = _count_patches((D, H, W))
        done = 0

        def update():
            nonlocal done, total
            done += 1
            with JobsLock:
                if job_id in Jobs:
                    Jobs[job_id]["progress"] = min(100.0, 100.0 * done / max(1, total))

        # Inline copy of infer loop with update callback
        runtime = get_runtime()
        pD, pH, pW = PATCH
        probe = np.zeros((1, stacked.shape[0], pD, pH, pW), dtype=np.float32)
        out = runtime.infer(probe)
        if out.ndim != 5:
            raise RuntimeError(f"Unexpected model output shape {out.shape}, expected [1, K, d, h, w]")
        K = out.shape[1]
        logits_sum = np.zeros((K, D, H, W), dtype=np.float32)
        weights_sum = np.zeros((D, H, W), dtype=np.float32)

        for z, y, x in sliding_window_coords((D, H, W), PATCH, STRIDE):
            patch = stacked[:, z:z+pD, y:y+pH, x:x+pW]
            pad_z = pD - patch.shape[1]
            pad_y = pH - patch.shape[2]
            pad_x = pW - patch.shape[3]
            if pad_z > 0 or pad_y > 0 or pad_x > 0:
                patch = np.pad(patch, ((0,0),(0,pad_z),(0,pad_y),(0,pad_x)), mode='constant')
            patch = patch[None, ...].astype(np.float32)
            logits = runtime.infer(patch)[0]
            logits = logits[:, :min(pD, D - z), :min(pH, H - y), :min(pW, W - x)]
            stitch_logits(logits_sum, weights_sum, logits, z, y, x)
            update()

        weights_sum = np.clip(weights_sum, 1e-6, None)
        logits_avg = logits_sum / weights_sum[None, ...]
        seg = np.argmax(logits_avg, axis=0).astype(np.int16)
        # Debug: summarize results
        try:
            uniq = np.unique(seg)
            print(f"[job {job_id}] completed. seg labels: {uniq.tolist()}")
        except Exception:
            pass

        with JobsLock:
            if job_id in Jobs:
                Jobs[job_id]["status"] = "done"
                Jobs[job_id]["progress"] = 100.0
                # Package as base64 npz for async as well
                def _b64(arr: np.ndarray, key: str) -> str:
                    buf = io.BytesIO()
                    np.savez_compressed(buf, **{key: arr})
                    return base64.b64encode(buf.getvalue()).decode('utf-8')
                Jobs[job_id]["result"] = {
                    "seg_npz": _b64(seg.astype(np.int16), 'seg'),
                    "meta": {
                        "labels": {"0": "background", "1": "edema", "2": "necrotic_core", "3": "enhancing"},
                        "unique_labels": sorted([int(x) for x in np.unique(seg)]),
                        "shape": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
                        "output_shapes": {
                            "seg": (int(seg.shape[0]), int(seg.shape[1]), int(seg.shape[2])),
                        },
                        "channel_order": ["T1", "T1ce", "T2", "FLAIR"],
                    }
                }
    except Exception as e:
        with JobsLock:
            if job_id in Jobs:
                Jobs[job_id]["status"] = "error"
                Jobs[job_id]["error"] = str(e)


@app.post("/infer_async")
def infer_async(req: InferenceRequest):
    try:
        np_vols: Dict[str, np.ndarray]
        mask_np: Optional[np.ndarray] = None
        spacing = req.spacing
        if req.npz_base64:
            raw = base64.b64decode(req.npz_base64)
            with np.load(io.BytesIO(raw), allow_pickle=False) as npz:
                np_vols = {}
                for k in ["T1", "T1ce", "T2", "FLAIR"]:
                    if k in npz:
                        np_vols[k] = np.array(npz[k], dtype=np.float32)
                if 'mask' in npz:
                    mask_np = np.array(npz['mask']).astype(np.uint8)
                if 'spacing' in npz and spacing is None:
                    spacing_arr = np.array(npz['spacing']).astype(np.float32).tolist()
                    if isinstance(spacing_arr, list) and len(spacing_arr) == 3:
                        spacing = (float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2]))
        else:
            if not req.volumes:
                raise HTTPException(status_code=400, detail="Provide either npz_base64 or volumes")
            np_vols = {k: np.asarray(v, dtype=np.float32) for k, v in req.volumes.items()}
            if req.mask is not None:
                mask_np = np.asarray(req.mask, dtype=np.uint8)
        shapes = {k: v.shape for k, v in np_vols.items()}
        if len(np_vols) == 0:
            raise HTTPException(status_code=400, detail="No volumes provided")
        if len({tuple(s) for s in shapes.values()}) != 1:
            raise HTTPException(status_code=400, detail=f"All volumes must share same shape, got: {shapes}")
        # Optional resample
        np_vols, _ = _resample_if_needed(np_vols, mask_np, spacing, req.target_spacing)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid volumes: {e}")

    job_id = uuid.uuid4().hex
    try:
        print(f"/infer_async -> job_id {job_id}, modalities: {list(np_vols.keys())}, shapes: {shapes}")
    except Exception:
        pass
    with JobsLock:
        Jobs[job_id] = {"status": "running", "progress": 0.0}
    t = threading.Thread(target=_infer_job, args=(job_id, np_vols), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    with JobsLock:
        if job_id not in Jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = Jobs[job_id]
        return {"status": j.get("status"), "progress": j.get("progress", 0.0), "error": j.get("error")}


@app.get("/result/{job_id}")
def get_result(job_id: str):
    with JobsLock:
        if job_id not in Jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        j = Jobs[job_id]
        if j.get("status") == "error":
            raise HTTPException(status_code=500, detail=j.get("error", "Unknown error"))
        if j.get("status") != "done":
            return {"status": j.get("status"), "progress": j.get("progress", 0.0)}
        return j["result"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
