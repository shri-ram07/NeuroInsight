import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { X, Box } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

interface ThreeDViewerProps {
  onClose: () => void;
  modality?: string;
  modalityImages?: { [modality: string]: any };
  segmentation?: number[][][] | null;
  labelNames?: Record<string, string>;
  visibleLabels?: { [k:number]: boolean };
  setVisibleLabels?: (v: { [k:number]: boolean }) => void;
  gradcamHeatmaps?: { [className: string]: number[][][] } | null;
  showGradcam?: boolean;
}

function getTypedArrayFromNifti(header: any, imageBuffer: ArrayBuffer):
  | Uint8Array | Int8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array {
  const code = header?.datatypeCode;
  switch (code) {
    case 2: return new Uint8Array(imageBuffer);
    case 256: return new Int8Array(imageBuffer);
    case 4: return new Int16Array(imageBuffer);
    case 512: return new Uint16Array(imageBuffer);
    case 8: return new Int32Array(imageBuffer);
    case 768: return new Uint32Array(imageBuffer);
    case 16: return new Float32Array(imageBuffer);
    case 64: return new Float64Array(imageBuffer);
    default: return new Float32Array(imageBuffer);
  }
}

// Stack masked slices (base) with optional segmentation overlays, background fully transparent
function StackSliceGroup({ data, dims, seg, gap = 0, filter = 'nearest', overlay = true, baseAlpha = 0.12, overlayAlpha = 0.7, maskBias = 0, visibleLabels }: {
  data: ArrayLike<number>;
  dims: [number, number, number];
  seg?: number[][][] | null;
  gap?: number;
  filter?: 'nearest' | 'linear';
  overlay?: boolean;
  baseAlpha?: number;
  overlayAlpha?: number;
  maskBias?: number; // -0.3..0.3 shifts Otsu threshold
  visibleLabels?: { [k:number]: boolean };
}) {
  const [D, H, W] = dims;
  const aspect = H / W;

  const segDims = useMemo(() => {
    if (!seg || !seg.length) return null;
    const d = seg.length; const h = seg[0]?.length || 0; const w = seg[0]?.[0]?.length || 0;
    if (!d || !h || !w) return null;
    return [d, h, w] as [number, number, number];
  }, [seg]);

  const slices = useMemo(() => {
    const list: { base: THREE.DataTexture; overlay?: THREE.DataTexture; z: number }[] = [];
    const bins = 256;
    for (let z = 0; z < D; z++) {
      const baseIdx = z * W * H;
      // min/max
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < W * H; i++) { const v = (data as any)[baseIdx + i]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (!Number.isFinite(mn) || !Number.isFinite(mx) || mx <= mn) continue;
      // normalize + hist
      const gray = new Uint8Array(W * H); const hist = new Uint32Array(bins);
      for (let i = 0; i < W * H; i++) { let t = ((data as any)[baseIdx + i] - mn) / (mx - mn); if (!Number.isFinite(t)) t = 0; if (t < 0) t = 0; if (t > 1) t = 1; const g = Math.floor(t * 255); gray[i] = g; hist[g]++; }
      // Otsu threshold
      let sum = 0; for (let t = 0; t < bins; t++) sum += t * hist[t];
      let sumB = 0, wB = 0, maxVar = 0, thr = 0; const total = W * H;
      for (let t = 0; t < bins; t++) { wB += hist[t]; if (wB === 0) continue; const wF = total - wB; if (wF === 0) break; sumB += t * hist[t]; const mB = sumB / wB; const mF = (sum - sumB) / wF; const between = wB * wF * (mB - mF) * (mB - mF); if (between > maxVar) { maxVar = between; thr = t; } }
  // base rgba (masked)
      const rgba = new Uint8Array(W * H * 4); let p = 0; const aBase = Math.floor(Math.max(0, Math.min(1, baseAlpha)) * 255);
  const thrAdj = Math.max(0, Math.min(255, Math.floor(thr + maskBias * 255)));
  for (let i = 0; i < W * H; i++) { const g = gray[i]; rgba[p++] = g; rgba[p++] = g; rgba[p++] = g; rgba[p++] = g <= thrAdj ? 0 : aBase; }
      const baseTex = new THREE.DataTexture(rgba, W, H); baseTex.format = THREE.RGBAFormat; baseTex.type = THREE.UnsignedByteType;
      baseTex.magFilter = filter === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter; baseTex.minFilter = filter === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter;
      // @ts-ignore
      baseTex.colorSpace = (THREE as any).SRGBColorSpace || 'srgb'; baseTex.unpackAlignment = 1; baseTex.needsUpdate = true;

      let overlayTex: THREE.DataTexture | undefined;
      if (overlay && seg && segDims) {
        const [SD, SH, SW] = segDims; const hasZ = z < SD;
        const rgbaO = new Uint8Array(W * H * 4); let q = 0; const a = Math.floor(Math.max(0, Math.min(1, overlayAlpha)) * 255);
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            const sx = Math.round((x / (W - 1)) * (SW - 1));
            const sy = Math.round((y / (H - 1)) * (SH - 1));
            let lab = hasZ ? Number(seg[z]?.[sy]?.[sx] ?? 0) : 0;
            if (lab === 3) lab = 4; // map ET alias
            let r = 0, g = 0, b = 0, aa = 0;
            if (lab !== 0) {
              if (visibleLabels && visibleLabels.hasOwnProperty(lab) && !visibleLabels[lab]) {
                // hidden label
              } else {
              if (lab === 1) { r = 25; g = 118; b = 210; aa = a; } // NCR/NET (dark blue)
              else if (lab === 2) { r = 56; g = 142; b = 60; aa = a; } // ED (dark green)
              else if (lab === 4) { r = 211; g = 47; b = 47; aa = a; } // ET (dark red)
              else { r = 189; g = 189; b = 189; aa = a; }
              // mask by base gray threshold
              const gv = gray[y * W + x]; if (gv <= thrAdj) aa = 0;
              }
            }
            rgbaO[q++] = r; rgbaO[q++] = g; rgbaO[q++] = b; rgbaO[q++] = aa;
          }
        }
        overlayTex = new THREE.DataTexture(rgbaO, W, H); overlayTex.format = THREE.RGBAFormat; overlayTex.type = THREE.UnsignedByteType;
        overlayTex.magFilter = filter === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter; overlayTex.minFilter = filter === 'nearest' ? THREE.NearestFilter : THREE.LinearFilter;
        overlayTex.unpackAlignment = 1; overlayTex.needsUpdate = true;
      }
      const zPos = ((z + 0.5) / D - 0.5) + (z - (D - 1) / 2) * gap;
      list.push({ base: baseTex, overlay: overlayTex, z: zPos });
    }
    return list;
  }, [data, D, H, W, seg, segDims, gap, filter, baseAlpha, overlayAlpha, overlay, visibleLabels, maskBias]);

  useEffect(() => () => { for (const s of slices) { s.base.dispose(); if (s.overlay) s.overlay.dispose(); } }, [slices]);

  return (
    <group>
      {slices.map((s, i) => (
        <group key={i} position={[0, 0, s.z]}>
          <mesh>
            <planeGeometry args={[1, aspect]} />
            <meshBasicMaterial map={s.base} transparent depthWrite={false} side={THREE.DoubleSide} />
          </mesh>
          {s.overlay && (
            <mesh>
              <planeGeometry args={[1, aspect]} />
              <meshBasicMaterial map={s.overlay} transparent depthWrite={false} side={THREE.DoubleSide} />
            </mesh>
          )}
        </group>
      ))}
    </group>
  );
}

export const ThreeDViewer = (props: ThreeDViewerProps) => {
  const { onClose, modality = 'T1', modalityImages = {}, segmentation, visibleLabels, setVisibleLabels, labelNames = {} } = props;
  const imageData = modalityImages?.[modality];
  const [showOverlay, setShowOverlay] = useState(true);
  const [gap, setGap] = useState(0);
  const [filterNearest, setFilterNearest] = useState(true);
  const [baseOpacity, setBaseOpacity] = useState(0.12);
  const [maskBias, setMaskBias] = useState(0);

  const dims: [number, number, number] | null = useMemo(() => {
    if (imageData?.header?.dims) { const d = imageData.header.dims; return [d[3] || 1, d[2] || 1, d[1] || 1]; }
    if (segmentation) return [segmentation.length, segmentation[0]?.length || 1, segmentation[0]?.[0]?.length || 1];
    return null;
  }, [imageData, segmentation]);

  const inputArray = useMemo(() => {
    if (!imageData?.image || !imageData?.header) return null;
    if (imageData.image instanceof ArrayBuffer) return getTypedArrayFromNifti(imageData.header, imageData.image);
    return ArrayBuffer.isView(imageData.image) ? (imageData.image as ArrayLike<number>) : null;
  }, [imageData]);

  const hasData = !!(inputArray && dims);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-2 md:p-4">
      <Card className="w-full max-w-6xl h-full max-h-[90vh] flex flex-col">
        <CardHeader className="flex flex-row items-center justify-between py-4">
          <CardTitle className="flex items-center gap-2">
            <Box className="h-5 w-5 text-primary" />
            3D Slice Stack
            {modality && (
              <span className="ml-2 text-xs text-muted-foreground">({modality})</span>
            )}
          </CardTitle>
          <div className="flex items-center gap-2 flex-wrap md:flex-nowrap md:overflow-visible overflow-x-auto max-w-full">
            {/* Per-label visibility toggles */}
            <div className="hidden md:flex items-center gap-3 text-xs bg-black/40 px-2 py-1 rounded">
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={!!visibleLabels?.[1]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 1: e.target.checked })} />
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(25,118,210)' }} />{labelNames['1'] || 'NCR/NET (1)'}
                </span>
              </label>
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={!!visibleLabels?.[2]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 2: e.target.checked })} />
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(56,142,60)' }} />{labelNames['2'] || 'Edema (2)'}</span>
              </label>
              <label className="flex items-center gap-1">
                <input type="checkbox" checked={!!visibleLabels?.[4]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 4: e.target.checked })} />
                <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(211,47,47)' }} />{labelNames['4'] || 'Enhancing (4)'}</span>
              </label>
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={showOverlay} onChange={(e)=> setShowOverlay(e.target.checked)} /> Overlay
            </label>
            <label className="flex items-center gap-2 text-sm whitespace-nowrap">
              Sharp <input type="checkbox" checked={filterNearest} onChange={(e)=> setFilterNearest(e.target.checked)} />
            </label>
            <div className="flex items-center gap-2 text-sm whitespace-nowrap">
              Gap <input className="w-24" type="range" min={0} max={0.02} step={0.001} value={gap} onChange={(e)=> setGap(parseFloat(e.target.value))} />
            </div>
            <div className="flex items-center gap-2 text-sm whitespace-nowrap">
              Base Opacity <input className="w-24" type="range" min={0.04} max={0.3} step={0.01} value={baseOpacity} onChange={(e)=> setBaseOpacity(parseFloat(e.target.value))} />
            </div>
            <div className="flex items-center gap-2 text-sm whitespace-nowrap">
              Mask Bias <input className="w-24" type="range" min={-0.3} max={0.3} step={0.01} value={maskBias} onChange={(e)=> setMaskBias(parseFloat(e.target.value))} />
            </div>
            <Button variant="outline" size="sm" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </CardHeader>

        <CardContent className="flex-1 p-0 bg-black">
          <div className="h-full rounded-b-lg relative bg-black">
            {!hasData ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-8">
                <div className="text-xl font-bold text-destructive mb-2">No Modality Data</div>
                <div className="text-muted-foreground">Upload modality volumes and try again.</div>
              </div>
            ) : (
              <Canvas camera={{ position: [1.5, 1.5, 1.5], fov: 50 }} gl={{ alpha: false, antialias: true }} style={{ background: 'black' }} onCreated={({ gl }) => { try { gl.setClearColor(0x000000, 1); } catch {} }}>
                <ambientLight intensity={0.6} />
                <directionalLight position={[2, 2, 2]} intensity={0.8} />
                <directionalLight position={[-2, -1, -2]} intensity={0.4} />
                {inputArray && dims && (
                  <StackSliceGroup
                    data={inputArray}
                    dims={dims}
                    seg={showOverlay ? segmentation : null}
                    gap={gap}
                    filter={filterNearest ? 'nearest' : 'linear'}
                    overlay={showOverlay}
                    baseAlpha={baseOpacity}
                    overlayAlpha={0.7}
                    maskBias={maskBias}
                    visibleLabels={visibleLabels}
                  />
                )}
                <OrbitControls makeDefault enablePan enableZoom enableRotate />
              </Canvas>
            )}

            {/* 3D Controls Info */}
            <div className="absolute bottom-4 left-4 bg-black/50 text-white p-3 rounded-lg text-sm space-y-1">
              <div>🖱️ Click & drag to rotate</div>
              <div>🖱️ Right-click & drag to pan</div>
              <div>🔄 Scroll to zoom</div>
            </div>
            {/* Scale overlay */}
            {imageData?.header && (
              <div className="absolute bottom-4 right-4 bg-black/60 text-white p-2 rounded-md text-xs">
                {(() => {
                  const pd = imageData.header.pixDims || imageData.header.pixdims || [];
                  const dx = Math.abs(pd[1] || 1), dy = Math.abs(pd[2] || 1), dz = Math.abs(pd[3] || 1);
                  const W = imageData.header.dims?.[1] || 0; const H = imageData.header.dims?.[2] || 0; const D = imageData.header.dims?.[3] || 0;
                  const sx = Math.round(W * dx), sy = Math.round(H * dy), sz = Math.round(D * dz);
                  const barMm = 50; // 50 mm reference bar
                  const barPx = 120; // fixed UI length for display
                  return (
                    <div className="space-y-1">
                      <div className="opacity-80">Voxel: {dx.toFixed(2)}×{dy.toFixed(2)}×{dz.toFixed(2)} mm</div>
                      <div className="opacity-80">Size: {sx}×{sy}×{sz} mm</div>
                      <div className="flex items-center gap-2 mt-1">
                        <div className="h-[2px] bg-white" style={{ width: barPx }} />
                        <span>{barMm} mm</span>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};