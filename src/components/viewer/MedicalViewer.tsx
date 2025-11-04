import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { RotateCcw, ZoomIn, ZoomOut, Move, Maximize2, RefreshCw, Download } from "lucide-react";
import { FileUploadZone } from "./FileUploadZone";

interface MedicalViewerProps {
  modality?: string;
  modalityImages?: { [modality: string]: any };
  setModalityImages?: (images: { [modality: string]: any }) => void;
  // overlays
  segmentation?: number[][][] | null;
  showSegmentation?: boolean;
  segmentationOpacity?: number[];
  visibleLabels?: { [k:number]: boolean };
  // Grad-CAM explainability
  gradcamHeatmaps?: { [className: string]: number[][][] } | null;
  showGradcam?: boolean;
  gradcamOpacity?: number[];
  uncertaintyMap?: number[][][] | null;
  showUncertainty?: boolean;
  uncertaintyOpacity?: number[];
  // labels/meta for legend
  labelNames?: Record<string, string>;
  presentLabels?: number[];
  // wiring / control
  onAnalyze?: () => Promise<void> | void;
  analyzing?: boolean;
  viewMode?: 'input' | 'output';
  onToggleView?: () => void;
  outputDims?: [number, number, number] | null;
  allOutputDims?: Record<string, [number, number, number]>;
}

// Map NIfTI datatype codes to TypedArray constructors
function getTypedArrayFromNifti(header: any, imageBuffer: ArrayBuffer):
  | Uint8Array
  | Int8Array
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array {
  const code = header?.datatypeCode;
  switch (code) {
    // See nifti-reader-js NIFTI1.TYPE_* codes
    case 2: // UINT8
      return new Uint8Array(imageBuffer);
    case 256: // INT8
      return new Int8Array(imageBuffer);
    case 4: // INT16
      return new Int16Array(imageBuffer);
    case 512: // UINT16
      return new Uint16Array(imageBuffer);
    case 8: // INT32
      return new Int32Array(imageBuffer);
    case 768: // UINT32
      return new Uint32Array(imageBuffer);
    case 16: // FLOAT32
      return new Float32Array(imageBuffer);
    case 64: // FLOAT64
      return new Float64Array(imageBuffer);
    default:
      // Fallback to Float32 if unknown
      return new Float32Array(imageBuffer);
  }
}

export const MedicalViewer = (props: MedicalViewerProps) => {
  const { modality = "T1", modalityImages = {}, setModalityImages } = props;
  const { segmentation, showSegmentation, segmentationOpacity, visibleLabels, labelNames = {}, presentLabels = [] } = props;
  const { gradcamHeatmaps, showGradcam = false, gradcamOpacity = [0.6] } = props;
  const { uncertaintyMap, showUncertainty = false, uncertaintyOpacity = [0.5] } = props;
  const { onAnalyze, analyzing = false, viewMode = 'input', onToggleView, outputDims, allOutputDims = {} } = props;

  const [currentSlice, setCurrentSlice] = useState<number[]>([1]);
  const [hasFiles, setHasFiles] = useState(false);
  const [zoom, setZoom] = useState(100);

  // Current modality data
  const imageData = modalityImages[modality];

  // Output presence detection
  const outputOnlyRaw = viewMode === 'output';

  const { isNifti, width, height, depth } = useMemo(() => {
    if (imageData?.header && imageData?.image) {
      const dims = imageData.header?.dims || [];
      return { isNifti: true, width: dims[1] || 0, height: dims[2] || 0, depth: Math.max(1, dims[3] || 1) };
    }
    if (imageData?.dataSet) return { isNifti: false, width: 0, height: 0, depth: 1 };
    return { isNifti: false, width: 0, height: 0, depth: 1 };
  }, [imageData]);

  const navDepth = Math.max(1, depth);

  useEffect(() => {
    setCurrentSlice((prev) => [Math.min(Math.max(prev[0] || 1, 1), navDepth)]);
  }, [navDepth]);

  // Grad-CAM heatmap overlay drawing function
  const drawGradCAM = useCallback((ctx: CanvasRenderingContext2D, dx: number, dy: number, destW: number, destH: number) => {
    if (!showGradcam || !gradcamHeatmaps) return;

    const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), Math.max((depth || 1) - 1, 0));
    const alpha = gradcamOpacity[0] ?? 0.6;

    let combinedHeatmap: number[][] | null = null;
    let maxIntensity = 0;

    Object.values(gradcamHeatmaps).forEach((heatmap) => {
      if (!heatmap || sliceIndex >= heatmap.length) return;
      const slice = heatmap[sliceIndex];
      if (!slice || !slice.length || !slice[0]?.length) return;

      const sliceHeight = slice.length;
      const sliceWidth = slice[0].length;
      if (!combinedHeatmap) {
        combinedHeatmap = Array.from({ length: sliceHeight }, () => Array(sliceWidth).fill(0));
      }
      if (!combinedHeatmap || combinedHeatmap.length !== sliceHeight || combinedHeatmap[0]?.length !== sliceWidth) {
        return;
      }
      for (let y = 0; y < sliceHeight; y++) {
        for (let x = 0; x < sliceWidth; x++) {
          const value = Number(slice[y]?.[x] ?? 0);
          combinedHeatmap[y][x] += value;
          if (combinedHeatmap[y][x] > maxIntensity) {
            maxIntensity = combinedHeatmap[y][x];
          }
        }
      }
    });

    if (!combinedHeatmap || maxIntensity <= 0) return;

    const height = combinedHeatmap.length;
    const width = combinedHeatmap[0].length;
    const heatmapCanvas = document.createElement('canvas');
    heatmapCanvas.width = width;
    heatmapCanvas.height = height;
    const heatCtx = heatmapCanvas.getContext('2d');
    if (!heatCtx) return;

    const imageData = heatCtx.createImageData(width, height);
    const getJetColor = (value: number): [number, number, number] => {
      const clipped = Math.max(0, Math.min(1, value));
      if (clipped < 0.25) return [0, Math.floor(255 * (clipped / 0.25)), 255];
      if (clipped < 0.5) return [0, 255, Math.floor(255 * (1 - (clipped - 0.25) / 0.25))];
      if (clipped < 0.75) return [Math.floor(255 * ((clipped - 0.5) / 0.25)), 255, 0];
      return [255, Math.floor(255 * (1 - (clipped - 0.75) / 0.25)), 0];
    };

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const normalized = combinedHeatmap[y][x] / maxIntensity;
        const [r, g, b] = getJetColor(normalized);
        const idx = (y * width + x) * 4;
        if (normalized > 0.001) {
          imageData.data[idx] = r;
          imageData.data[idx + 1] = g;
          imageData.data[idx + 2] = b;
          imageData.data[idx + 3] = Math.floor(255 * alpha * Math.max(0.3, normalized));
        } else {
          imageData.data[idx + 3] = 0;
        }
      }
    }

    heatCtx.putImageData(imageData, 0, 0);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.drawImage(heatmapCanvas, 0, 0, width, height, dx, dy, destW, destH);
    ctx.restore();

    if (import.meta.env.DEV) {
      console.debug('Grad-CAM overlay drawn', { sliceIndex, keys: Object.keys(gradcamHeatmaps) });
    }
  }, [showGradcam, gradcamHeatmaps, gradcamOpacity, currentSlice, depth]);

  const drawUncertainty = useCallback((ctx: CanvasRenderingContext2D, dx: number, dy: number, destW: number, destH: number) => {
    if (!showUncertainty || !uncertaintyMap) return;
    const depthSlices = uncertaintyMap.length;
    if (!depthSlices) return;

    const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), depthSlices - 1);
    const slice = uncertaintyMap[sliceIndex];
    if (!slice || !slice[0]) return;

    const height = slice.length;
    const width = slice[0].length;
    if (!height || !width) return;

    let localMin = Number.POSITIVE_INFINITY;
    let localMax = Number.NEGATIVE_INFINITY;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const val = Number(slice[y]?.[x]);
        if (!Number.isFinite(val)) continue;
        if (val < localMin) localMin = val;
        if (val > localMax) localMax = val;
      }
    }

    if (!Number.isFinite(localMin) || !Number.isFinite(localMax) || localMax <= localMin) {
      return;
    }

    const alpha = uncertaintyOpacity[0] ?? 0.5;
    const heatCanvas = document.createElement('canvas');
    heatCanvas.width = width;
    heatCanvas.height = height;
    const heatCtx = heatCanvas.getContext('2d');
    if (!heatCtx) return;

    const imageData = heatCtx.createImageData(width, height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const raw = Number(slice[y]?.[x] ?? 0);
        if (!Number.isFinite(raw)) continue;
        let normalized = (raw - localMin) / (localMax - localMin);
        if (!Number.isFinite(normalized)) normalized = 0;
        normalized = Math.max(0, Math.min(1, normalized));
        const idx = (y * width + x) * 4;
        const intensity = normalized ** 0.55; // keep subtle gradients but boost hotspots
        const r = 255;
        const g = Math.floor(140 * (1 - intensity));
        const b = Math.floor(32 * (1 - intensity));
        imageData.data[idx] = r;
        imageData.data[idx + 1] = g;
        imageData.data[idx + 2] = b;
        imageData.data[idx + 3] = Math.floor(255 * alpha * intensity);
      }
    }

    heatCtx.putImageData(imageData, 0, 0);
    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.drawImage(heatCanvas, 0, 0, width, height, dx, dy, destW, destH);
    ctx.restore();
  }, [showUncertainty, uncertaintyMap, uncertaintyOpacity, currentSlice]);

  // Refs
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);

  // Optional: auto-jump slice when not frozen
  // No auto-jump for output; we render only input view pixels

  // Base MRI render — do nothing when frozen (keeps exact pixels on screen)
  useEffect(() => {
    if (outputOnlyRaw) return; // don't render input when in output-only view
    if (!isNifti || !imageData?.header || !imageData?.image) return;

    const box = boxRef.current; const canvas = canvasRef.current; if (!canvas || !box) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;
    const w = width, h = height; if (!w || !h) return;

    const rect = box.getBoundingClientRect();
    const displayW = Math.round(rect.width), displayH = Math.round(rect.height);
    canvas.width = displayW; canvas.height = displayH;
    // @ts-ignore
    ctx.imageSmoothingEnabled = false; ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,displayW,displayH);

    let dataView: any = imageData.image;
    if (dataView instanceof ArrayBuffer) dataView = getTypedArrayFromNifti(imageData.header, dataView);
    if (!(ArrayBuffer.isView(dataView))) { try { dataView = new Float32Array(dataView); } catch { return; } }

    const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), Math.max(depth - 1, 0));
    const sliceOffset = sliceIndex * w * h;
    let min = Infinity, max = -Infinity; const wh = w*h;
    for (let i = 0; i < wh; i++) { const v = (dataView as any)[sliceOffset + i]; if (v < min) min = v; if (v > max) max = v; }
    if (!isFinite(min) || !isFinite(max) || min === max) return;

    const rgba = new Uint8ClampedArray(wh * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const idx = x + y * w + sliceOffset; let val = (dataView as any)[idx]; val = ((val - min) / (max - min)) * 255; val = Math.max(0, Math.min(255, val));
      const p = (y * w + x) * 4; rgba[p] = val; rgba[p+1] = val; rgba[p+2] = val; rgba[p+3] = 255;
    }
    const imgData = new ImageData(rgba, w, h);

    const off = document.createElement('canvas'); off.width = w; off.height = h; const oc = off.getContext('2d');
    if (oc) {
      // @ts-ignore
      oc.imageSmoothingEnabled = false; oc.imageSmoothingQuality = 'high'; oc.putImageData(imgData, 0, 0);
      const scale = Math.min(displayW / w, displayH / h);
      const destW = Math.max(1, Math.floor(w * scale));
      const destH = Math.max(1, Math.floor(h * scale));
      const dx = Math.floor((displayW - destW) / 2);
      const dy = Math.floor((displayH - destH) / 2);
      ctx.clearRect(0, 0, displayW, displayH);
      ctx.drawImage(off, 0, 0, w, h, dx, dy, destW, destH);
      if (destW > 0 && destH > 0) {
        drawGradCAM(ctx, dx, dy, destW, destH);
        drawUncertainty(ctx, dx, dy, destW, destH);
      }
    }
  }, [isNifti, imageData, width, height, depth, currentSlice, modality, outputOnlyRaw, drawGradCAM, drawUncertainty]);

  // Output rendering: draw base grayscale slice and segmentation overlay (output-only view)
  useEffect(() => {
    if (!outputOnlyRaw) return; // Only render in output view
    const box = boxRef.current; const canvas = canvasRef.current; if (!canvas || !box) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;

    // Dimensions for base input (if available)
    const W_in = width, H_in = height;

    // Dimensions for segmentation (if available)
    const D = segmentation?.length || 0;
    const H_seg = segmentation && segmentation[0] ? (segmentation[0].length || 0) : 0;
    const W_seg = segmentation && segmentation[0] && segmentation[0][0] ? (segmentation[0][0].length || 0) : 0;

    const rect = box.getBoundingClientRect();
    const displayW = Math.round(rect.width), displayH = Math.round(rect.height);
    canvas.width = displayW; canvas.height = displayH;
    // @ts-ignore
    ctx.imageSmoothingEnabled = false; ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,displayW,displayH);

    // Compute a common draw area using input dims when present, else seg dims
    const baseW = W_in || W_seg || 1;
    const baseH = H_in || H_seg || 1;
    const scale = Math.min(displayW / baseW, displayH / baseH);
    const destW = Math.max(1, Math.floor(baseW * scale));
    const destH = Math.max(1, Math.floor(baseH * scale));
    const dx = Math.floor((displayW - destW) / 2);
    const dy = Math.floor((displayH - destH) / 2);

    // Draw base grayscale slice from current modality if we have NIfTI data
    // Also compute a mask threshold to remove black background when overlaying
    if (isNifti && imageData?.header && imageData?.image && W_in && H_in) {
      let dataView: any = imageData.image;
      if (dataView instanceof ArrayBuffer) dataView = getTypedArrayFromNifti(imageData.header, dataView);
      if (ArrayBuffer.isView(dataView)) {
        const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), Math.max((depth || 1) - 1, 0));
        const sliceOffset = sliceIndex * W_in * H_in;
        let min = Infinity, max = -Infinity; const wh = W_in * H_in;
        for (let i = 0; i < wh; i++) { const v = (dataView as any)[sliceOffset + i]; if (v < min) min = v; if (v > max) max = v; }
        if (isFinite(min) && isFinite(max) && min !== max) {
          // Build normalized grayscale and histogram for Otsu threshold
          const gray = new Uint8Array(wh);
          const hist = new Uint32Array(256);
          for (let i = 0; i < wh; i++) {
            let val = (dataView as any)[sliceOffset + i];
            val = ((val - min) / (max - min)); if (!Number.isFinite(val)) val = 0; if (val < 0) val = 0; if (val > 1) val = 1;
            const g = Math.floor(val * 255);
            gray[i] = g; hist[g]++;
          }
          // Otsu threshold (0..255)
          let sum = 0; for (let t = 0; t < 256; t++) sum += t * hist[t];
          let sumB = 0; let wB = 0; let wF = 0; let maxVar = 0; let thr = 0;
          const total = wh;
          for (let t = 0; t < 256; t++) {
            wB += hist[t]; if (wB === 0) continue; wF = total - wB; if (wF === 0) break;
            sumB += t * hist[t];
            const mB = sumB / wB; const mF = (sum - sumB) / wF;
            const between = wB * wF * (mB - mF) * (mB - mF);
            if (between > maxVar) { maxVar = between; thr = t; }
          }
          // Build RGBA for base and mask background: pixels <= thr are fully transparent
          const rgba = new Uint8ClampedArray(wh * 4);
          for (let i = 0; i < wh; i++) {
            const v = gray[i]; const p = i * 4; rgba[p] = v; rgba[p+1] = v; rgba[p+2] = v; rgba[p+3] = v <= thr ? 0 : 255;
          }
          const imgData = new ImageData(rgba, W_in, H_in);
          const off = document.createElement('canvas'); off.width = W_in; off.height = H_in; const oc = off.getContext('2d');
          if (oc) { oc.putImageData(imgData, 0, 0); ctx.drawImage(off, 0, 0, W_in, H_in, dx, dy, destW, destH); }
          // Save threshold in closure for overlay composition
          ;(ctx as any)._brainMaskThreshold = thr;
        }
      }
    } else {
      // Fallback background
      ctx.clearRect(0, 0, displayW, displayH); // allow transparent background
    }

    // If segmentation not available or hidden, stop after base
    if (!segmentation || !showSegmentation) return;

    // Build segmentation color image for the current slice
    if (!D || !H_seg || !W_seg) return;
    const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), Math.max(D - 1, 0));
    const segZ = segmentation[sliceIndex]; if (!segZ) return;

    const img = new ImageData(W_seg, H_seg);
    let anyLabel = false;
    const a = Math.round(255 * (segmentationOpacity?.[0] ?? 0.6));
    const color = (labelIn: number): [number, number, number, number] => {
      // Normalize BraTS labels: background 0, NCR/NET 1, ED 2, ET 4. Some models use 3 for ET, map 3->4
      const label = (labelIn === 3 ? 4 : labelIn);
      if (label === 0) return [0, 0, 0, 0];
      // Darker RGB palette for stronger contrast
      if (label === 1) return [25, 118, 210, a];   // blue 600 (NCR/NET)
      if (label === 2) return [56, 142, 60, a];    // green 700 (ED)
      if (label === 4) return [211, 47, 47, a];    // red 700 (ET)
      const palette: Array<[number, number, number]> = [[206, 147, 216], [255, 224, 130], [255, 204, 128], [197, 225, 165], [179, 229, 252]];
      const [r, g, b] = palette[label % palette.length];
      return [r, g, b, a];
    };
    // If we drew a base slice earlier, retrieve the threshold to mask background
    const thr = (ctx as any)._brainMaskThreshold as number | undefined;
    const haveMask = typeof thr === 'number' && Number.isFinite(thr);
    // Precompute mapping between dest and seg coords for mask sampling
    // We'll mask by checking the base intensity (after scaling) corresponding to each seg pixel
    let maskLookup: Uint8Array | null = null;
    if (haveMask && isNifti && W_in && H_in && imageData?.image) {
      // Rebuild the normalized gray (same as above but only for current slice)
      let dataView: any = imageData.image; if (dataView instanceof ArrayBuffer) dataView = getTypedArrayFromNifti(imageData.header, dataView);
      const sliceIndex = Math.min(Math.max((currentSlice[0] || 1) - 1, 0), Math.max((depth || 1) - 1, 0));
      const sliceOffset = sliceIndex * W_in * H_in; let min = Infinity, max = -Infinity; const wh = W_in * H_in;
      for (let i = 0; i < wh; i++) { const v = (dataView as any)[sliceOffset + i]; if (v < min) min = v; if (v > max) max = v; }
      if (isFinite(min) && isFinite(max) && min !== max) {
        maskLookup = new Uint8Array(wh);
        for (let i = 0; i < wh; i++) {
          let val = (dataView as any)[sliceOffset + i]; val = ((val - min) / (max - min)); if (!Number.isFinite(val)) val = 0; if (val < 0) val = 0; if (val > 1) val = 1;
          maskLookup[i] = Math.floor(val * 255);
        }
      }
    }
    for (let y = 0; y < H_seg; y++) {
      for (let x = 0; x < W_seg; x++) {
        let lab = Number(segZ[y]?.[x] ?? 0);
        if (lab === 3) lab = 4; // map ET alias
        const p = (y * W_seg + x) * 4;
        let [r, g, b, aa] = color(lab);
        // Hide disabled labels
        if (lab !== 0 && visibleLabels && visibleLabels.hasOwnProperty(lab) && !visibleLabels[lab]) {
          aa = 0;
        }
        // Apply mask: if base intensity under this seg pixel is below threshold, force transparent
        if (haveMask && maskLookup) {
          // Map seg (x,y) to base (x',y') assuming both are scaled to same dest area dx,dy,destW,destH
          // Compute normalized coords in dest and back to base indices
          const nx = x / (W_seg - 1); const ny = y / (H_seg - 1);
          const bx = Math.min(W_in - 1, Math.max(0, Math.round(nx * (W_in - 1))));
          const by = Math.min(H_in - 1, Math.max(0, Math.round(ny * (H_in - 1))));
          const gv = maskLookup[by * W_in + bx];
          if (gv <= (thr ?? 0)) aa = 0;
        }
        if (lab !== 0 && aa > 0) anyLabel = true;
        img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = aa;
      }
    }
    const off = document.createElement('canvas'); off.width = W_seg; off.height = H_seg; off.getContext('2d')!.putImageData(img, 0, 0);
    // Draw overlay scaled to the same area as base for alignment
    ctx.drawImage(off, 0, 0, W_seg, H_seg, dx, dy, destW, destH);

    // Draw Grad-CAM heatmaps on top of segmentation
    drawGradCAM(ctx, dx, dy, destW, destH);

  // Draw uncertainty overlay if enabled
  drawUncertainty(ctx, dx, dy, destW, destH);

    // Friendly hint when a slice has no labels
    if (!anyLabel) {
      ctx.save();
      ctx.fillStyle = 'rgba(255,255,255,0.75)';
      ctx.font = '12px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial';
      ctx.fillText('No segmented voxels on this slice', dx + 12, dy + destH - 12);
      ctx.restore();
    }
  }, [outputOnlyRaw, segmentation, segmentationOpacity, currentSlice, isNifti, imageData, width, height, depth, drawGradCAM, drawUncertainty]);



  const handleFilesUploaded = () => setHasFiles(true);
  const handleAnalyze = async () => { if (onAnalyze) await onAnalyze(); };

  if (!hasFiles) {
    return (
      <div className="flex-1 p-6">
        <FileUploadZone onFilesUploaded={handleFilesUploaded} setModalityImages={setModalityImages} />
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col">
      <div className="p-4 border-b border-border bg-card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {viewMode === 'input' && (
              <>
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium">Slice:</Label>
                  <Input type="number" value={currentSlice[0]} onChange={(e) => setCurrentSlice([Math.max(1, Math.min(parseInt(e.target.value) || 1, navDepth))])} className="w-20 h-8" min={1} max={navDepth} />
                  <span className="text-sm text-muted-foreground">/ {navDepth}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Label className="text-sm font-medium">Zoom:</Label>
                  <Badge variant="outline">{zoom}%</Badge>
                </div>
              </>
            )}
          </div>
          <div className="flex items-center gap-2">
            {viewMode === 'input' && (
              <>
                <Button variant="outline" size="sm" onClick={() => setZoom((z) => Math.max(10, z - 10))}><ZoomOut className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm" onClick={() => setZoom((z) => Math.min(400, z + 10))}><ZoomIn className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm"><Move className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm" onClick={() => setCurrentSlice([1])}><RotateCcw className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm" onClick={() => setZoom(100)}><RefreshCw className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm"><Maximize2 className="h-4 w-4" /></Button>
                <Button variant="outline" size="sm"><Download className="h-4 w-4" /></Button>
              </>
            )}
            <Button variant="secondary" size="sm" onClick={onToggleView}>{viewMode === 'input' ? 'Show Output Only' : 'Show Input Only'}</Button>
            <Button variant="default" size="sm" onClick={handleAnalyze} disabled={analyzing}>{analyzing ? 'Analyzing…' : 'Analyze Scan'}</Button>
          </div>
        </div>
      </div>

      <div className="flex-1 p-6">
        <div className="h-full medical-viewer relative flex items-center justify-center">
          <div ref={boxRef} className="w-full h-full max-w-[600px] max-h-[600px] rounded-lg relative">
            {viewMode === 'input' ? (
              <div className={`absolute inset-0 flex items-center justify-center ${analyzing ? 'analyze-blur' : ''}`}>
                <canvas ref={canvasRef} style={{ display: 'block', background: 'black', borderRadius: 16, maxWidth: '100%', maxHeight: '100%', width: 'auto', height: 'auto', objectFit: 'contain' }} />
              </div>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center">
                <canvas ref={canvasRef} style={{ display: 'block', background: 'transparent', borderRadius: 16, maxWidth: '100%', maxHeight: '100%', width: 'auto', height: 'auto', objectFit: 'contain' }} />
              </div>
            )}

            {analyzing && (
              <div className="analyze-loader">
                <div style={{ position: 'relative', width: 160, height: 120 }}>
                  <div className="Strich1">
                    <div className="Strich2">
                      <div className="bubble" />
                      <div className="bubble1" />
                      <div className="bubble2" />
                      <div className="bubble3" />
                      <div className="bubble4" />
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="absolute top-4 left-4 space-y-1">
              {viewMode === 'input' ? (
                <>
                  <Badge className="bg-black/50 text-white">{`${modality}-weighted`}</Badge>
                  <div className="text-white text-xs bg-black/50 px-2 py-1 rounded">Slice {currentSlice[0]} / {navDepth}</div>
                </>
              ) : (
                <>
                  <Badge className="bg-black/50 text-white">Segmented Output</Badge>
                  <div className="text-white text-xs bg-black/50 px-2 py-1 rounded">Slice {currentSlice[0]} / {segmentation?.length || navDepth}</div>
                </>
              )}
            </div>
            <div className="absolute bottom-4 right-4 text-white text-xs bg-black/50 px-2 py-1 rounded">{viewMode === 'input' ? 'Axial View' : 'Output view'}</div>
            {viewMode === 'output' && showSegmentation && (
              <div className="absolute bottom-20 left-4 bg-black/60 text-white text-xs px-2 py-2 rounded-md shadow space-y-1 max-w-[180px]">
                <div className="font-semibold text-[11px]">Legend</div>
                <div className="flex flex-col gap-1">
                  {[1,2,4]
                    .filter((l) => presentLabels.includes(l) || (l === 4 && presentLabels.includes(3)))
                    .map((lab) => {
                      const names: Record<number,string> = {
                        1: 'NCR/NET (Necrotic & non‑enhancing core)',
                        2: 'ED (Peritumoral edema)',
                        4: 'ET (Enhancing tumor)'
                      };
                      const swatch = lab === 1 ? 'rgb(25,118,210)' : lab === 2 ? 'rgb(56,142,60)' : 'rgb(211,47,47)';
                      const labelText = names[lab] || `Label ${lab}`; // enforce canonical BraTS names in 2D legend
                      return (
                        <div key={lab} className="flex items-center gap-2 leading-tight">
                          <div className="w-2.5 h-2.5 rounded-sm border border-white/30" style={{ backgroundColor: swatch }} />
                          <span className="text-[11px]">{labelText} ({lab})</span>
                        </div>
                      );
                    })}
                </div>
                {showGradcam && gradcamHeatmaps && (
                  <div className="border-t border-white/20 pt-1 mt-1">
                    <div className="font-semibold text-yellow-300 text-[11px]">Explainability</div>
                    <div className="text-[10px] opacity-80">Grad-CAM heatmaps active</div>
                  </div>
                )}
                {showUncertainty && uncertaintyMap && (
                  <div className="border-t border-white/20 pt-1 mt-1">
                    <div className="font-semibold text-orange-300 text-[11px]">Uncertainty</div>
                    <div className="text-[10px] opacity-80">Warm colors = lower confidence</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {(viewMode === 'input' || viewMode === 'output') && (
        <div className="p-4 border-t border-border bg-card">
          <div className="max-w-2xl mx-auto">
            <Label className="block text-sm font-medium mb-2">Slice Navigation</Label>
            <Slider value={currentSlice} onValueChange={setCurrentSlice} max={outputOnlyRaw && segmentation ? segmentation.length : navDepth} min={1} step={1} className="w-full" />
            <div className="flex justify-between text-xs text-muted-foreground mt-1"><span>1</span><span>{outputOnlyRaw && segmentation ? segmentation.length : navDepth}</span></div>
          </div>
        </div>
      )}
    </div>
  );
};

