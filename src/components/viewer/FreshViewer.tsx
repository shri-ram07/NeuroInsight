import React, { useEffect, useMemo, useRef, useState } from "react";

// Minimal, clean viewer that renders exactly ONE centered frame with overlays.
// It avoids CSS scaling entirely by sizing canvases to the wrapper box
// and draws a single aspect-fit image. No legacy code paths.

export interface FreshViewerProps {
  // Base MRI (optional): dims inferred via header.dims [?, H, W, D]
  modality?: string;
  modalityImages?: { [modality: string]: any };

  // Outputs (voxel-space arrays at shape [D][H][W])
  segmentation?: number[][][] | null;
  heatmap?: number[][][] | null;
  uncertainty?: number[][][] | null;

  // Visibility + opacity controls
  showSegmentation?: boolean;
  segmentationOpacity?: number; // 0..1
  showGradCAM?: boolean;
  gradCAMOpacity?: number; // 0..1
  showUncertainty?: boolean;
  uncertaintyOpacity?: number; // 0..1
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

export const FreshViewer: React.FC<FreshViewerProps> = (props) => {
  const {
    modality = "T1",
    modalityImages = {},
    segmentation,
    heatmap,
    uncertainty,
    showSegmentation = true,
    segmentationOpacity = 0.7,
    showGradCAM = false,
    gradCAMOpacity = 0.8,
    showUncertainty = false,
    uncertaintyOpacity = 0.6,
  } = props;

  const imageData = modalityImages[modality];

  // Outputs
  const outD = segmentation?.length || heatmap?.length || uncertainty?.length || 0;
  const one = segmentation ?? heatmap ?? uncertainty ?? null;
  const outH = one && one[0] ? one[0].length : 0;
  const outW = one && one[0] && one[0][0] ? one[0][0].length : 0;
  const outputOnly = !!one;

  const { isNifti, width, height, depth } = useMemo(() => {
    if (imageData?.header && imageData?.image) {
      const dims = imageData.header?.dims || [];
      return { isNifti: true, width: dims[1] || 0, height: dims[2] || 0, depth: Math.max(1, dims[3] || 1) };
    }
    return { isNifti: false, width: 0, height: 0, depth: 1 };
  }, [imageData]);

  const navDepth = outputOnly ? Math.max(outD, 1) : Math.max(depth, 1);
  const [sliceIdx, setSliceIdx] = useState(1);
  useEffect(() => {
    setSliceIdx((s) => Math.min(Math.max(s, 1), navDepth));
  }, [navDepth]);

  const boxRef = useRef<HTMLDivElement | null>(null);
  const baseRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);

  // Base MRI
  useEffect(() => {
    if (outputOnly) return; // skip base if outputs present
    if (!isNifti || !imageData?.header || !imageData?.image) return;
    const box = boxRef.current, canvas = baseRef.current; if (!box || !canvas) return;
    const ctx = canvas.getContext('2d'); if (!ctx) return;

    const W = width, H = height; if (!W || !H) return;

    const rectW = Math.max(1, box.clientWidth), rectH = Math.max(1, box.clientHeight);
    canvas.width = rectW; canvas.height = rectH;
    // @ts-ignore
    ctx.imageSmoothingEnabled = false; ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0, 0, rectW, rectH);

    let dataView: any = imageData.image;
    if (dataView instanceof ArrayBuffer) dataView = getTypedArrayFromNifti(imageData.header, dataView);
    if (!(ArrayBuffer.isView(dataView))) { try { dataView = new Float32Array(dataView);} catch { return; } }

    const z = Math.min(Math.max(sliceIdx - 1, 0), Math.max(depth - 1, 0));
    const offset = z * W * H;

    let min = Infinity, max = -Infinity; const WH = W * H;
    for (let i=0;i<WH;i++){const v=(dataView as any)[offset + i]; if(v<min) min=v; if(v>max) max=v;}
    if (!isFinite(min) || !isFinite(max) || min===max) return;

    const rgba = new Uint8ClampedArray(WH * 4);
    for (let y=0;y<H;y++) for (let x=0;x<W;x++){
      const idx = x + y*W + offset; let val=(dataView as any)[idx];
      val=((val-min)/(max-min))*255; val=Math.max(0,Math.min(255,val));
      const p=(y*W+x)*4; rgba[p]=val; rgba[p+1]=val; rgba[p+2]=val; rgba[p+3]=255;
    }
    const img = new ImageData(rgba, W, H);
    const off = document.createElement('canvas'); off.width=W; off.height=H;
    off.getContext('2d')!.putImageData(img, 0, 0);

    const scale = Math.min(rectW / W, rectH / H);
    const dW = Math.max(1, Math.floor(W*scale)), dH = Math.max(1, Math.floor(H*scale));
    const dx = Math.floor((rectW - dW)/2), dy = Math.floor((rectH - dH)/2);
    ctx.drawImage(off, 0, 0, W, H, dx, dy, dW, dH);
  }, [outputOnly, isNifti, imageData, width, height, depth, sliceIdx]);

  useEffect(() => {
    // overlays or output-only
    const box = boxRef.current, ov = overlayRef.current; if (!box || !ov) return;
    const ctx = ov.getContext('2d'); if (!ctx) return;

    const W = outW || width, H = outH || height; if (!W || !H) return;

    const rectW = Math.max(1, box.clientWidth), rectH = Math.max(1, box.clientHeight);
    ov.width = rectW; ov.height = rectH;
    // @ts-ignore
    ctx.imageSmoothingEnabled = false; ctx.setTransform(1,0,0,1,0,0);
    ctx.clearRect(0, 0, rectW, rectH);

    // Composite at voxel res
    const comp = document.createElement('canvas'); comp.width=W; comp.height=H;
    const cctx = comp.getContext('2d')!; // assume available
    // @ts-ignore
    cctx.imageSmoothingEnabled = false; cctx.setTransform(1,0,0,1,0,0);
    cctx.clearRect(0, 0, W, H);
    if (outputOnly) { cctx.fillStyle = 'black'; cctx.fillRect(0,0,W,H); }

    const read2D = (mat: any[][], y: number, x: number) => {
      const mh = mat?.length ?? 0; const mw = mh ? (mat[0]?.length ?? 0) : 0;
      if (!mh || !mw) return 0;
      const looksHW = (mh===H && mw===W); const looksWH = (mh===W && mw===H);
      if (looksHW && y<mh && x<mw) return mat[y][x];
      if (looksWH && x<mh && y<mw) return mat[x][y];
      if (y<mh && x<mw) return mat[y][x];
      return 0;
    };

    const scaleSlice = (mat: any[][]) => {
      const vals: number[] = []; for (let y=0;y<H;y++){ for (let x=0;x<W;x++){ const v=Number(read2D(mat,y,x)||0); if(Number.isFinite(v)) vals.push(v);} }
      if (!vals.length) return {a:0,b:1}; vals.sort((a,b)=>a-b);
      const q=(p:number)=> vals[Math.min(vals.length-1, Math.max(0, Math.floor((p/100)*vals.length)))];
      const a=q(2), b=q(98); const span=b-a; return {a, b: span>1e-6?b:a+1};
    };

    const viridis = (()=>{ const lut:number[][]=[]; for(let i=0;i<256;i++){ const t=i/255; const r=Math.round(255*Math.min(1,Math.max(0,0.267+2.42*t-4.74*t*t+2.73*t*t*t))); const g=Math.round(255*Math.min(1,Math.max(0,0.005+2.10*t-1.53*t*t+0.26*t*t*t))); const b=Math.round(255*Math.min(1,Math.max(0,0.334+0.89*t-0.55*t*t+0.15*t*t*t))); lut.push([r,g,b]); } return (v:number)=> lut[Math.min(255, Math.max(0, Math.floor(v*255)))] })();

    let anyOverlay=false;
    const z = Math.min(Math.max(sliceIdx-1,0), Math.max((outputOnly?outD:depth)-1,0));

    // Segmentation fill
    if (showSegmentation && segmentation) {
      const segZ = segmentation[z]; if (segZ && segZ.length && segZ[0].length) {
        const img = cctx.createImageData(W, H);
        const palette: Array<[number,number,number]> = [
          [55,126,184], [77,175,74], [228,26,28], [152,78,163], [255,127,0], [166,86,40], [247,129,191], [153,153,153]
        ];
        const colorFor=(label:number):[number,number,number,number]=>{
          if(label===0) return [0,0,0,0];
          if(label===1) return [55,126,184, Math.round(255*segmentationOpacity)];
          if(label===2) return [77,175,74, Math.round(255*segmentationOpacity)];
          if(label===3) return [228,26,28, Math.round(255*segmentationOpacity)];
          const [r,g,b]=palette[(label%palette.length+palette.length)%palette.length];
          return [r,g,b, Math.round(255*segmentationOpacity)];
        };
        for(let y=0;y<H;y++) for(let x=0;x<W;x++){
          const label=Number(read2D(segZ as any,y,x)||0); const p=(y*W+x)*4; const [r,g,b,a]=colorFor(label);
          if (a>0) anyOverlay=true; img.data[p]=r; img.data[p+1]=g; img.data[p+2]=b; img.data[p+3]=a;
        }
        const off = document.createElement('canvas'); off.width=W; off.height=H; off.getContext('2d')!.putImageData(img,0,0);
        cctx.drawImage(off,0,0);
      }
    }

    // Heatmap
    if (showGradCAM && heatmap) {
      const heatZ = heatmap[z]; if (heatZ && heatZ.length && heatZ[0].length){
        const {a,b}=scaleSlice(heatZ as any); const img=cctx.createImageData(W,H); const vals:number[]=[];
        for(let y=0;y<H;y++) for(let x=0;x<W;x++){
          const raw=Number(read2D(heatZ as any,y,x)||0); const v=Math.max(0,Math.min(1,(raw-a)/(b-a))); const [r,g,b_]=viridis(v); const p=(y*W+x)*4;
          img.data[p]=r; img.data[p+1]=g; img.data[p+2]=b_; img.data[p+3]=Math.round(255*gradCAMOpacity); if(v>0.01) anyOverlay=true; vals.push(v);
        }
        let off=document.createElement('canvas'); off.width=W; off.height=H; off.getContext('2d')!.putImageData(img,0,0); cctx.drawImage(off,0,0);
        if(vals.length){ vals.sort((m,n)=>m-n); const thr=vals[Math.max(0,Math.floor(vals.length*0.95))];
          const hi=cctx.createImageData(W,H); for(let y=0;y<H;y++) for(let x=0;x<W;x++){
            const raw=Number(read2D(heatZ as any,y,x)||0); const v=Math.max(0,Math.min(1,(raw-a)/(b-a))); if(v>=thr){ const p=(y*W+x)*4; hi.data[p]=255; hi.data[p+1]=235; hi.data[p+2]=59; hi.data[p+3]=Math.round(255*Math.min(1,gradCAMOpacity+0.2)); }
          }
          off=document.createElement('canvas'); off.width=W; off.height=H; off.getContext('2d')!.putImageData(hi,0,0); cctx.drawImage(off,0,0);
          const outline=cctx.createImageData(W,H); for(let y=0;y<H;y++) for(let x=0;x<W;x++){
            const raw=Number(read2D(heatZ as any,y,x)||0); const v=Math.max(0,Math.min(1,(raw-a)/(b-a))); if(v>=thr){
              const n1=Math.max(0,Math.min(1,(Number(read2D(heatZ as any,y,x+1)||0)-a)/(b-a)));
              const n2=Math.max(0,Math.min(1,(Number(read2D(heatZ as any,y+1,x)||0)-a)/(b-a)));
              if(n1<thr || n2<thr){ const p=(y*W+x)*4; outline.data[p]=255; outline.data[p+1]=255; outline.data[p+2]=255; outline.data[p+3]=200; }
            }
          }
          off=document.createElement('canvas'); off.width=W; off.height=H; off.getContext('2d')!.putImageData(outline,0,0); cctx.drawImage(off,0,0);
        }
      }
    }

    // Segmentation edges
    if (showSegmentation && segmentation) {
      const segZ = segmentation[z];
      if (segZ && segZ.length && segZ[0].length) {
        const edge = cctx.createImageData(W, H);
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const c = Number(read2D(segZ as any, y, x) || 0);
          if (c === 0) continue;
          const n1 = Number(read2D(segZ as any, y, x + 1) || c);
          const n2 = Number(read2D(segZ as any, y + 1, x) || c);
          if (n1 !== c || n2 !== c) { const p = (y * W + x) * 4; edge.data[p] = 255; edge.data[p + 1] = 255; edge.data[p + 2] = 255; edge.data[p + 3] = 200; }
        }
        const off = document.createElement('canvas'); off.width = W; off.height = H; off.getContext('2d')!.putImageData(edge, 0, 0); cctx.drawImage(off, 0, 0);
      }
    }

    // Uncertainty
    if (showUncertainty && uncertainty){ const uncZ=uncertainty[z]; if(uncZ && uncZ.length && uncZ[0].length){
      const {a,b}=scaleSlice(uncZ as any); const img=cctx.createImageData(W,H);
      for(let y=0;y<H;y++) for(let x=0;x<W;x++){
        const raw=Number(read2D(uncZ as any,y,x)||0); const v=Math.max(0,Math.min(1,(raw-a)/(b-a))); const [r,g,b_]=viridis(v); const p=(y*W+x)*4;
        img.data[p]=r; img.data[p+1]=g; img.data[p+2]=b_; img.data[p+3]=Math.round(255*uncertaintyOpacity);
      }
      const off=document.createElement('canvas'); off.width=W; off.height=H; off.getContext('2d')!.putImageData(img,0,0); cctx.drawImage(off,0,0);
    }}

    const scale = Math.min(rectW / W, rectH / H);
    const dW = Math.max(1, Math.floor(W*scale)), dH = Math.max(1, Math.floor(H*scale));
    const dx = Math.floor((rectW - dW)/2), dy = Math.floor((rectH - dH)/2);
    ctx.drawImage(comp, 0, 0, W, H, dx, dy, dW, dH);

    if (!anyOverlay){ ctx.save(); ctx.fillStyle='rgba(255,255,255,0.6)'; ctx.font='12px sans-serif'; ctx.fillText(outputOnly?'No output on this slice':'No overlay on this slice', 10, rectH-10); ctx.restore(); }
  }, [outputOnly, outD, outH, outW, width, height, depth, sliceIdx, segmentation, heatmap, uncertainty, showSegmentation, showGradCAM, showUncertainty, segmentationOpacity, gradCAMOpacity, uncertaintyOpacity]);

  return (
    <div className="w-full h-full flex flex-col">
      <div className="p-2 text-sm">Slice: {sliceIdx} / {navDepth}</div>
      <div ref={boxRef} className="relative w-full h-full max-w-[600px] max-h-[600px] bg-black rounded-lg overflow-hidden">
        <canvas ref={baseRef} style={{ display: outputOnly ? 'none' : 'block' }} />
        <canvas ref={overlayRef} style={{ position:'absolute', inset:0, pointerEvents:'none' }} />
      </div>
      <div className="p-2">
        <input type="range" min={1} max={navDepth} value={sliceIdx} onChange={(e)=>setSliceIdx(parseInt(e.target.value)||1)} />
      </div>
    </div>
  );
};
