import { useEffect, useRef, useState } from "react";
import { TopNavigation } from "@/components/layout/TopNavigation";
import { ControlSidebar } from "@/components/layout/ControlSidebar";
import { MedicalViewer } from "@/components/viewer/MedicalViewer";
import { StatusBar } from "@/components/layout/StatusBar";
import { decodeNpzTo3DArray } from "@/lib/npz";

type Volume3D = number[][][];

type Shape3D = [number, number, number];

const Index = () => {
	const [sidebarOpen, setSidebarOpen] = useState(true);
	const [selectedModality, setSelectedModality] = useState("T1");
	const [modalityImages, setModalityImages] = useState<Record<string, any>>({});

	const [segmentation, setSegmentation] = useState<Volume3D | null>(null);
	const [outputDims, setOutputDims] = useState<Shape3D | null>(null);
	const [allOutputDims, setAllOutputDims] = useState<Record<string, Shape3D>>({});
	const [labelNames, setLabelNames] = useState<Record<string, string>>({});
	const [presentLabels, setPresentLabels] = useState<number[]>([]);
	const [viewMode, setViewMode] = useState<"input" | "output">("input");
	const [showSegmentation, setShowSegmentation] = useState(false);
	const [segmentationOpacity, setSegmentationOpacity] = useState([0.7]);
	const [visibleLabels, setVisibleLabels] = useState<{ [k: number]: boolean }>(
		{ 1: true, 2: true, 3: true, 4: true }
	);
	const [analyzing, setAnalyzing] = useState(false);
	const [analysisProgress, setAnalysisProgress] = useState(0);

	const [explainabilityMode, setExplainabilityMode] = useState(false);
	const [gradcamHeatmaps, setGradcamHeatmaps] = useState<Record<string, Volume3D> | null>(null);
	const [showGradcam, setShowGradcam] = useState(false);
	const [gradcamOpacity, setGradcamOpacity] = useState([0.6]);
	const [uncertaintyMap, setUncertaintyMap] = useState<Volume3D | null>(null);
	const [showUncertainty, setShowUncertainty] = useState(false);
	const [uncertaintyOpacity, setUncertaintyOpacity] = useState([0.5]);

	useEffect(() => {
		if (!explainabilityMode) {
			setShowGradcam(false);
		} else if (gradcamHeatmaps) {
			setShowGradcam(true);
		}
	}, [explainabilityMode, gradcamHeatmaps]);

	const pollTimer = useRef<number | null>(null);
	const polling = useRef(false);

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
			case 2:
				return new Uint8Array(imageBuffer);
			case 256:
				return new Int8Array(imageBuffer);
			case 4:
				return new Int16Array(imageBuffer);
			case 512:
				return new Uint16Array(imageBuffer);
			case 8:
				return new Int32Array(imageBuffer);
			case 768:
				return new Uint32Array(imageBuffer);
			case 16:
				return new Float32Array(imageBuffer);
			case 64:
				return new Float64Array(imageBuffer);
			default:
				return new Float32Array(imageBuffer);
		}
	}

	function reshapeToDHW(v: any) {
		const dims = v.header.dims;
		const W = dims[1];
		const H = dims[2];
		const D = dims[3];
		let buffer: any = v.image;
		if (v.image instanceof ArrayBuffer) {
			buffer = getTypedArrayFromNifti(v.header, v.image);
		}
		const out: Volume3D = Array.from({ length: D }, () =>
			Array.from({ length: H }, () => Array(W).fill(0))
		);
		for (let z = 0; z < D; z++) {
			const base = z * W * H;
			for (let y = 0; y < H; y++) {
				for (let x = 0; x < W; x++) {
					const val = Number(buffer[base + y * W + x]);
					out[z][y][x] = Number.isFinite(val) ? val : 0;
				}
			}
		}
		return { out, shape: [D, H, W] as Shape3D };
	}

	function computeVolumeStats(volume: Volume3D) {
		let min = Number.POSITIVE_INFINITY;
		let max = Number.NEGATIVE_INFINITY;
		let sum = 0;
		let count = 0;

		for (const plane of volume) {
			for (const row of plane) {
				for (const value of row) {
					const num = Number(value);
					if (!Number.isFinite(num)) continue;
					if (num < min) min = num;
					if (num > max) max = num;
					sum += num;
					count += 1;
				}
			}
		}

		if (count === 0) {
			return { min: 0, max: 0, mean: 0 };
		}

		return { min, max, mean: sum / count };
	}

	async function analyzeVolumes() {
		try {
			const order = ["T1", "T1ce", "T2", "FLAIR"] as const;
			const available = order.filter((m) => modalityImages[m]?.header && modalityImages[m]?.image);
			if (available.length < 4) {
				alert("Please upload all four modalities: T1, T1ce, T2, and FLAIR.");
				return;
			}

			const volumes: Record<string, Volume3D> = {};
			let refShape: Shape3D | null = null;
			for (const modality of available) {
				const v = modalityImages[modality];
				const { out, shape } = reshapeToDHW(v);
				if (!refShape) {
					refShape = shape;
				} else if (refShape[0] !== shape[0] || refShape[1] !== shape[1] || refShape[2] !== shape[2]) {
					throw new Error(
						`Volume shape mismatch: ${modality} has [${shape.join(', ')}], expected [${refShape.join(', ')}]`
					);
				}
				volumes[modality] = out;
			}

			const apiBase = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

			setGradcamHeatmaps(null);
			setShowGradcam(false);
			setUncertaintyMap(null);
			setShowUncertainty(false);

			if (explainabilityMode) {
				setAnalyzing(true);
				setAnalysisProgress(50);

				const response = await fetch(`${apiBase}/infer`, {
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify({ volumes, enable_gradcam: true }),
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

				const shape = (data?.meta?.shape || []) as number[];
				const outputShapes = (data?.meta?.output_shapes || {}) as Record<string, number[]>;

				if (Array.isArray(shape) && shape.length === 3) {
					setOutputDims([shape[0] || 0, shape[1] || 0, shape[2] || 0]);
					const mapped: Record<string, Shape3D> = {};
					for (const [key, value] of Object.entries(outputShapes)) {
						if (Array.isArray(value) && value.length === 3) {
							mapped[key] = [value[0] || 0, value[1] || 0, value[2] || 0];
						}
					}
					setAllOutputDims(mapped);
					if (data?.meta?.labels && typeof data.meta.labels === "object") {
						setLabelNames(data.meta.labels);
					}
					if (Array.isArray(data?.meta?.unique_labels)) {
						setPresentLabels(
							data.meta.unique_labels
								.filter((n: any) => Number.isFinite(n))
								.map((n: any) => Number(n))
						);
					}

					try {
						const segArr = await decodeNpzTo3DArray(data?.seg_npz, "seg");
						setSegmentation(segArr);
						setShowSegmentation(true);
						setViewMode("output");
					} catch (segErr) {
						console.error("Failed to decode seg_npz", segErr);
					}

					if (data?.gradcam_npz) {
						try {
							const gradcamData: Record<string, Volume3D> = {};
							const perClassKeys: Array<{ key: string; label: string }> = [
								{ key: "gradcam_necrotic", label: "necrotic" },
								{ key: "gradcam_edema", label: "edema" },
								{ key: "gradcam_enhancing", label: "enhancing" },
							];

							for (const { key, label } of perClassKeys) {
								try {
									const map = await decodeNpzTo3DArray(data.gradcam_npz, key, { quietMissing: true });
									gradcamData[label] = map;
								} catch {
									// per-class heatmap missing is acceptable
								}
							}

							if (Object.keys(gradcamData).length === 0) {
								try {
									const focusMap = await decodeNpzTo3DArray(data.gradcam_npz, "gradcam_focus", { quietMissing: true });
									const roiClass = data?.meta?.gradcam_roi?.class_name;
									const label = typeof roiClass === "string" && roiClass.length > 0 ? roiClass : "focus";
									gradcamData[label] = focusMap;
								} catch {
									console.warn("Grad-CAM response did not include known heatmap keys.");
								}
							}

							if (Object.keys(gradcamData).length > 0) {
								setGradcamHeatmaps(gradcamData);
								setShowGradcam(true);
							}
						} catch (camErr) {
							console.error("Failed to decode Grad-CAM data", camErr);
							setGradcamHeatmaps(null);
							setShowGradcam(false);
						}
					}

					if (data?.aux_npz) {
						try {
							const uncert = await decodeNpzTo3DArray(data.aux_npz, "uncertainty_map");
							const stats = computeVolumeStats(uncert);
							const hasSignal = stats.max > 0.01;
							setUncertaintyMap(uncert);
							setShowUncertainty(hasSignal);
							if (hasSignal && (uncertaintyOpacity?.[0] ?? 0) <= 0) {
								setUncertaintyOpacity([0.5]);
							}
						} catch (uncertErr) {
							console.warn("Failed to decode uncertainty map", uncertErr);
							setUncertaintyMap(null);
							setShowUncertainty(false);
						}
					}
				}

				return;
			}

			const start = await fetch(`${apiBase}/infer_async`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ volumes }),
			});

			if (!start.ok) {
				const text = await start.text();
				try {
					const j = JSON.parse(text);
					throw new Error(j.detail || text);
				} catch {
					throw new Error(text || `Start failed: ${start.status}`);
				}
			}

			const startJson = await start.json();
			const { job_id } = startJson;
			setAnalyzing(true);
			setAnalysisProgress(0);

			await new Promise<void>((resolve, reject) => {
				let delay = 800;
				let notFoundGraceTries = 4;
				let finished = false;

				const tick = async () => {
					if (polling.current || finished) return;
					polling.current = true;
					try {
						const pr = await fetch(`${apiBase}/progress/${job_id}`);
						if (!pr.ok) {
							const t = await pr.text();
							if (pr.status === 404 && /Job not found/i.test(t) && notFoundGraceTries > 0) {
								notFoundGraceTries -= 1;
								return;
							}
							try {
								const j = JSON.parse(t);
								throw new Error(j.detail || t);
							} catch {
								throw new Error(t || `Progress failed: ${pr.status}`);
							}
						}

						const pj = await pr.json();
						if (typeof pj.progress === "number") {
							setAnalysisProgress(pj.progress);
						}

						if (pj.status === "error") {
							throw new Error(pj.error || "Job failed");
						}

						if (pj.status === "done") {
							const rr = await fetch(`${apiBase}/result/${job_id}`);
							if (!rr.ok) {
								const t = await rr.text();
								try {
									const j = JSON.parse(t);
									throw new Error(j.detail || t);
								} catch {
									throw new Error(t || `Result failed: ${rr.status}`);
								}
							}

							const data = await rr.json();
							const shape = (data?.meta?.shape || []) as number[];
							const outputShapes = (data?.meta?.output_shapes || {}) as Record<string, number[]>;

							if (Array.isArray(shape) && shape.length === 3) {
								setOutputDims([shape[0] || 0, shape[1] || 0, shape[2] || 0]);
								const mapped: Record<string, Shape3D> = {};
								for (const [key, value] of Object.entries(outputShapes)) {
									if (Array.isArray(value) && value.length === 3) {
										mapped[key] = [value[0] || 0, value[1] || 0, value[2] || 0];
									}
								}
								setAllOutputDims(mapped);
								if (data?.meta?.labels && typeof data.meta.labels === "object") {
									setLabelNames(data.meta.labels);
								}
								if (Array.isArray(data?.meta?.unique_labels)) {
									setPresentLabels(
										data.meta.unique_labels
											.filter((n: any) => Number.isFinite(n))
											.map((n: any) => Number(n))
									);
								}
								try {
									const segArr = await decodeNpzTo3DArray(data?.seg_npz, "seg");
									setSegmentation(segArr);
									setShowSegmentation(true);
									setViewMode("output");
									setAnalysisProgress(100);
								} catch (segErr) {
									console.error("Failed to decode seg_npz", segErr);
								}
							}

							if (data?.aux_npz) {
								try {
									const uncert = await decodeNpzTo3DArray(data.aux_npz, "uncertainty_map");
									const stats = computeVolumeStats(uncert);
									const hasSignal = stats.max > 0.01;
									setUncertaintyMap(uncert);
									setShowUncertainty(hasSignal);
									if (hasSignal && (uncertaintyOpacity?.[0] ?? 0) <= 0) {
										setUncertaintyOpacity([0.5]);
									}
								} catch (uncertErr) {
									console.warn("Failed to decode uncertainty map", uncertErr);
									setUncertaintyMap(null);
									setShowUncertainty(false);
								}
							}

							finished = true;
							resolve();
						}
					} catch (err) {
						finished = true;
						reject(err);
					} finally {
						polling.current = false;
						if (!finished) {
							delay = Math.min(2000, delay + 200);
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
			polling.current = false;
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
			<TopNavigation onMenuToggle={() => setSidebarOpen(!sidebarOpen)} />

			<div className="flex-1 flex">
				<div
					className={`transition-all duration-300 ease-in-out ${sidebarOpen ? "w-80" : "w-0"} md:relative absolute inset-y-0 left-0 z-40 overflow-auto`}
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
							uncertaintyMap={uncertaintyMap}
							showUncertainty={showUncertainty}
							setShowUncertainty={setShowUncertainty}
							uncertaintyOpacity={uncertaintyOpacity}
							setUncertaintyOpacity={setUncertaintyOpacity}
						/>
					)}
				</div>

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
						uncertaintyMap={uncertaintyMap}
						showUncertainty={showUncertainty}
						uncertaintyOpacity={uncertaintyOpacity}
						labelNames={labelNames}
						presentLabels={presentLabels}
						visibleLabels={visibleLabels}
						onAnalyze={analyzeVolumes}
						analyzing={analyzing}
						viewMode={viewMode}
						onToggleView={() => setViewMode((mode) => (mode === "input" ? "output" : "input"))}
						outputDims={outputDims}
						allOutputDims={allOutputDims}
					/>
				</div>
			</div>

			<StatusBar analyzing={analyzing} progress={analysisProgress} />
		</div>
	);
};

export default Index;
