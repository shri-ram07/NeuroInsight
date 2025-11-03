import { useEffect, useRef, useState } from "react";
import { TopNavigation } from "@/components/layout/TopNavigation";
import { ControlSidebar } from "@/components/layout/ControlSidebar";
import { MedicalViewer } from "@/components/viewer/MedicalViewer";
import { StatusBar } from "@/components/layout/StatusBar";
import { decodeNpzTo3DArray } from "@/lib/npz";

const Index = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedModality, setSelectedModality] = useState("T1");
  const [modalityImages, setModalityImages] = useState<{ [modality: string]: any }>({});

  // Analysis results and toggles
  const [segmentation, setSegmentation] = useState<number[][][] | null>(null);
  const [outputDims, setOutputDims] = useState<[number, number, number] | null>(null);
  const [allOutputDims, setAllOutputDims] = useState<Record<string, [number, number, number]>>({});
  const [labelNames, setLabelNames] = useState<Record<string, string>>({});
  const [presentLabels, setPresentLabels] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<'input' | 'output'>('input');
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [segmentationOpacity, setSegmentationOpacity] = useState([0.7]);
  // Visible label toggles follow BraTS classes: 1 (NCR/NET), 2 (ED), 4 (ET)
  // Keep 3 enabled for compatibility if some models output ET as 3
  const [visibleLabels, setVisibleLabels] = useState<{[k:number]: boolean}>({1:true,2:true,4:true,3:true});
  const [analyzing, setAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  
  // Explainability Mode and Grad-CAM
  const [explainabilityMode, setExplainabilityMode] = useState(false);
  const [gradcamHeatmaps, setGradcamHeatmaps] = useState<{ [className: string]: number[][][] } | null>(null);
  const [showGradcam, setShowGradcam] = useState(false);
  const [gradcamOpacity, setGradcamOpacity] = useState([0.6]);
  
  // Sync Grad-CAM visibility with explainability mode
  useEffect(() => {
    if (!explainabilityMode) {
      // When explainability mode is turned off, hide Grad-CAM
      setShowGradcam(false);
    } else if (explainabilityMode && gradcamHeatmaps) {
      // When explainability mode is turned on and we have data, show Grad-CAM
      setShowGradcam(true);
    }
  }, [explainabilityMode, gradcamHeatmaps]);
  const pollTimer = useRef<number | null>(null);
  const polling = useRef<boolean>(false);

  // Map NIfTI datatype to TypedArray for correct backend inputs
  function getTypedArrayFromNifti(header: any, imageBuffer: ArrayBuffer):
    | Uint8Array | Int8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array {
    const code = header?.datatypeCode;
    switch (code) {
      case 2: return new Uint8Array(imageBuffer);      // UINT8
      case 256: return new Int8Array(imageBuffer);     // INT8
      case 4: return new Int16Array(imageBuffer);      // INT16
      case 512: return new Uint16Array(imageBuffer);   // UINT16
      case 8: return new Int32Array(imageBuffer);      // INT32
      case 768: return new Uint32Array(imageBuffer);   // UINT32
      case 16: return new Float32Array(imageBuffer);   // FLOAT32
      case 64: return new Float64Array(imageBuffer);   // FLOAT64
      default: return new Float32Array(imageBuffer);
    }
  }

  function reshapeToDHW(v: any) {
    const dims = v.header.dims; // [ndim, X, Y, Z, ...]
    const W = dims[1], H = dims[2], D = dims[3];
    let buffer: any = v.image;
    if (v.image instanceof ArrayBuffer) {
      buffer = getTypedArrayFromNifti(v.header, v.image);
    }
    const out: number[][][] = Array.from({ length: D }, () => Array.from({ length: H }, () => Array(W).fill(0)));
    for (let z = 0; z < D; z++) {
      const base = z * W * H;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const val = Number(buffer[base + y * W + x]);
          out[z][y][x] = Number.isFinite(val) ? val : 0;
        }
      }
    }
    return { out, shape: [D, H, W] as [number, number, number] };
  }

  async function analyzeVolumes() {
    try {
      const order = ["T1", "T1ce", "T2", "FLAIR"] as const;
      const available = order.filter((m) => modalityImages[m]?.header && modalityImages[m]?.image);
      if (available.length < 4) {
        alert("Please upload all four modalities: T1, T1ce, T2, and FLAIR.");
        return;
      }
      // Build volumes and ensure shapes match
      const volumes: Record<string, number[][][]> = {};
      let refShape: [number, number, number] | null = null;
      for (const m of available) {
        const v = modalityImages[m];
        const { out, shape } = reshapeToDHW(v);
        if (!refShape) refShape = shape;
        if (refShape[0] !== shape[0] || refShape[1] !== shape[1] || refShape[2] !== shape[2]) {
          throw new Error(`Volume shape mismatch: ${m} has [${shape.join(', ')}], expected [${refShape.join(', ')}]`);
        }
        volumes[m] = out;
      }
      try { console.log("Analyze request volumes:", Object.fromEntries(Object.entries(volumes).map(([k,v]) => [k, { D: v.length, H: v[0]?.length, W: v[0]?.[0]?.length }]))); } catch {}

      const apiBase = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";
      
      // Debug: Log explainability mode status
      console.log("Explainability mode:", explainabilityMode);
      
      // Use synchronous endpoint for explainability mode, async for regular inference
      if (explainabilityMode) {
        console.log("Using Grad-CAM explainability mode - synchronous inference");
        setAnalyzing(true);
        setAnalysisProgress(50);
        
        const response = await fetch(`${apiBase}/infer`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            volumes,
            enable_gradcam: true
          })
        });
        
        if (!response.ok) {
          const text = await response.text();
          try { 
            const j = JSON.parse(text); 
            throw new Error(j.detail || text); 
          } catch { 
            throw new Error(text || `Inference failed: ${response.status}`); 
          }
        }
        
        const data = await response.json();
        setAnalysisProgress(100);
        
        // Process results (same as async path)
        const shape = (data?.meta?.shape || []) as number[];
        const outputShapes = (data?.meta?.output_shapes || {}) as Record<string, number[]>;
        
        if (Array.isArray(shape) && shape.length === 3) {
          setOutputDims([shape[0] || 0, shape[1] || 0, shape[2] || 0]);
          const mapped: Record<string, [number, number, number]> = {};
          for (const [k, v] of Object.entries(outputShapes)) {
            if (Array.isArray(v) && v.length === 3) mapped[k] = [v[0] || 0, v[1] || 0, v[2] || 0];
          }
          setAllOutputDims(mapped);
          if (data?.meta?.labels && typeof data.meta.labels === 'object') setLabelNames(data.meta.labels);
          if (Array.isArray(data?.meta?.unique_labels)) setPresentLabels(data.meta.unique_labels.filter((n: any) => Number.isFinite(n)).map((n: any) => Number(n)));
          
          // Decode segmentation
          try {
            const segArr = await decodeNpzTo3DArray(data?.seg_npz, 'seg');
            setSegmentation(segArr);
            setShowSegmentation(true);
            setViewMode('output');
          } catch (e) {
            console.error('Failed to decode seg_npz', e);
          }
          
          // Decode Grad-CAM heatmaps if available
          if (data?.gradcam_npz) {
            console.log('Received Grad-CAM NPZ data, decoding...');
            try {
              const gradcamData: { [className: string]: number[][][] } = {};
              for (const className of ['necrotic', 'edema', 'enhancing']) {
                try {
                  const heatmap = await decodeNpzTo3DArray(data.gradcam_npz, `gradcam_${className}`);
                  gradcamData[className] = heatmap;
                  console.log(`Decoded Grad-CAM for ${className}:`, heatmap ? `${heatmap.length}x${heatmap[0]?.length}x${heatmap[0]?.[0]?.length}` : 'null');
                } catch (e) {
                  console.warn(`Failed to decode Grad-CAM for ${className}:`, e);
                }
              }
              console.log('Setting Grad-CAM data:', Object.keys(gradcamData));
              setGradcamHeatmaps(gradcamData);
              setShowGradcam(true);
            } catch (e) {
              console.error('Failed to decode Grad-CAM data:', e);
            }
          } else {
            console.log('No Grad-CAM data received in response');
          }
        }
        
        return; // Exit early for explainability mode
      }
      
      // Regular async inference
      const start = await fetch(`${apiBase}/infer_async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ volumes })
      });
      if (!start.ok) {
        const text = await start.text();
        try { const j = JSON.parse(text); throw new Error(j.detail || text); } catch { throw new Error(text || `Start failed: ${start.status}`); }
      }
  const startJson = await start.json();
  console.log("infer_async start response:", startJson);
  const { job_id } = startJson;
      setAnalyzing(true);
      setAnalysisProgress(0);

      await new Promise<void>((resolve, reject) => {
        let delay = 800; // start 0.8s, backoff to 2s
        let notFoundGraceTries = 4; // tolerate a few 404s if backend just started the job
        let finished = false;
        const tick = async () => {
          if (polling.current) return; // prevent overlaps
          polling.current = true;
          try {
            const pr = await fetch(`${apiBase}/progress/${job_id}`);
            if (!pr.ok) {
              const t = await pr.text();
              // Gracefully handle transient 404 Job not found
              if (pr.status === 404 && /Job not found/i.test(t) && notFoundGraceTries > 0) {
                notFoundGraceTries -= 1;
                return; // fall through to finally -> schedule next tick
              }
              try { const j = JSON.parse(t); throw new Error(j.detail || t); } catch { throw new Error(t || `Progress failed: ${pr.status}`); }
            }
            const pj = await pr.json();
            try { console.log("progress:", pj); } catch {}
            if (typeof pj.progress === "number") setAnalysisProgress(pj.progress);
            if (pj.status === "error") throw new Error(pj.error || "Job failed");
            // Only fetch result when backend marks status === "done" (progress can reach 100 before results are ready)
            if (pj.status === "done") {
              const rr = await fetch(`${apiBase}/result/${job_id}`);
              if (!rr.ok) {
                const t = await rr.text();
                try { const j = JSON.parse(t); throw new Error(j.detail || t); } catch { throw new Error(t || `Result failed: ${rr.status}`); }
              }
              const data = await rr.json();
              // Read output dims and decode segmentation npz
              const shape = (data?.meta?.shape || []) as number[];
              const outputShapes = (data?.meta?.output_shapes || {}) as Record<string, number[]>;
              try { console.log('unique_labels:', data?.meta?.unique_labels); } catch {}
              try { console.log('labels:', data?.meta?.labels); } catch {}
              if (Array.isArray(shape) && shape.length === 3) {
                setOutputDims([shape[0] || 0, shape[1] || 0, shape[2] || 0]);
                const mapped: Record<string, [number, number, number]> = {};
                for (const [k, v] of Object.entries(outputShapes)) {
                  if (Array.isArray(v) && v.length === 3) mapped[k] = [v[0] || 0, v[1] || 0, v[2] || 0];
                }
                setAllOutputDims(mapped);
                if (data?.meta?.labels && typeof data.meta.labels === 'object') setLabelNames(data.meta.labels);
                if (Array.isArray(data?.meta?.unique_labels)) setPresentLabels(data.meta.unique_labels.filter((n: any) => Number.isFinite(n)).map((n: any) => Number(n)));
                // Decode seg_npz -> 3D int array
                try {
                  const segArr = await decodeNpzTo3DArray(data?.seg_npz, 'seg');
                  setSegmentation(segArr);
                  setShowSegmentation(true);
                  setViewMode('output');
                } catch (e) {
                  console.error('Failed to decode seg_npz', e);
                }
                finished = true;
                resolve();
                return;
              }
            }
            delay = Math.min(2000, delay + 200); // gentle backoff
          } catch (e) {
            finished = true; 
            reject(e);
            return;
          } finally {
            polling.current = false;
            if (!finished) {
              pollTimer.current = window.setTimeout(tick, delay);
            }
          }
        };
        tick();
      });
    } catch (err: any) {
  console.error("Analyze error:", err);
  alert(`Analyze error: ${err?.message || err}`);
    } finally {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
      setAnalyzing(false);
    }
  }

  useEffect(() => {
    return () => {
      if (pollTimer.current) {
        window.clearTimeout(pollTimer.current);
        pollTimer.current = null;
      }
    };
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background overflow-auto">
      {/* Top Navigation */}
      <TopNavigation onMenuToggle={() => setSidebarOpen(!sidebarOpen)} />

      {/* Main Content Area */}
      <div className="flex-1 flex">
        {/* Control Sidebar */}
        <div
          className={`transition-all duration-300 ease-in-out ${sidebarOpen ? 'w-80' : 'w-0'} md:relative absolute inset-y-0 left-0 z-40 overflow-auto`}
        >
          {sidebarOpen && (
            <ControlSidebar 
              selectedModality={selectedModality} 
              setSelectedModality={setSelectedModality}
              onAnalyze={analyzeVolumes}
              analyzing={analyzing}
              progress={analysisProgress}
              showSegmentation={showSegmentation}
              setShowSegmentation={setShowSegmentation}
              segmentationOpacity={segmentationOpacity}
              setSegmentationOpacity={setSegmentationOpacity}
              modalityImages={modalityImages}
              segmentation={segmentation}
              labelNames={labelNames}
              visibleLabels={visibleLabels}
              setVisibleLabels={setVisibleLabels}
              explainabilityMode={explainabilityMode}
              setExplainabilityMode={setExplainabilityMode}
              gradcamHeatmaps={gradcamHeatmaps}
              showGradcam={showGradcam}
              setShowGradcam={setShowGradcam}
              gradcamOpacity={gradcamOpacity}
              setGradcamOpacity={setGradcamOpacity}
            />
          )}
        </div>

        {/* Main Viewer */}
        <div className="flex-1 flex flex-col overflow-auto">
          <MedicalViewer
            modality={selectedModality}
            modalityImages={modalityImages}
            setModalityImages={setModalityImages}
            segmentation={segmentation}
            showSegmentation={showSegmentation}
            segmentationOpacity={segmentationOpacity}
            gradcamHeatmaps={gradcamHeatmaps}
            showGradcam={showGradcam}
            gradcamOpacity={gradcamOpacity}
            labelNames={labelNames}
            presentLabels={presentLabels}
            visibleLabels={visibleLabels}
            onAnalyze={analyzeVolumes}
            analyzing={analyzing}
            viewMode={viewMode}
            onToggleView={() => setViewMode((m) => (m === 'input' ? 'output' : 'input'))}
            outputDims={outputDims}
            allOutputDims={allOutputDims}
          />
        </div>
      </div>

      {/* Status Bar */}
  <StatusBar analyzing={analyzing} progress={analysisProgress} />
    </div>
  );
};

export default Index;
