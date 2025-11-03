import { Button } from "@/components/ui/button";
import { ThreeDViewer } from "@/components/viewer/ThreeDViewer";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import { Layers, Eye, Brain, Play, Volume2, Box, Lightbulb, Target } from "lucide-react";
import { useState } from "react";

interface ControlSidebarProps {
  selectedModality: string;
  setSelectedModality: (modality: string) => void;
  onAnalyze: () => Promise<void> | void;
  analyzing?: boolean;
  progress?: number;
  showSegmentation: boolean;
  setShowSegmentation: (v: boolean) => void;
  segmentationOpacity: number[];
  setSegmentationOpacity: (v: number[]) => void;
  modalityImages?: { [modality: string]: any };
  segmentation?: number[][][] | null;
  labelNames?: Record<string, string>;
  visibleLabels?: { [k:number]: boolean };
  setVisibleLabels?: (v: { [k:number]: boolean }) => void;
  explainabilityMode?: boolean;
  setExplainabilityMode?: (v: boolean) => void;
  gradcamHeatmaps?: { [className: string]: number[][][] } | null;
  showGradcam?: boolean;
  setShowGradcam?: (v: boolean) => void;
  gradcamOpacity?: number[];
  setGradcamOpacity?: (v: number[]) => void;
}

export const ControlSidebar = ({
  selectedModality,
  setSelectedModality,
  onAnalyze,
  analyzing = false,
  progress = 0,
  showSegmentation,
  setShowSegmentation,
  segmentationOpacity,
  setSegmentationOpacity,
  modalityImages,
  segmentation,
  labelNames,
  visibleLabels,
  setVisibleLabels,
  explainabilityMode = false,
  setExplainabilityMode,
  gradcamHeatmaps,
  showGradcam = false,
  setShowGradcam,
  gradcamOpacity = [0.6],
  setGradcamOpacity,
}: ControlSidebarProps) => {
  const [show3DViewer, setShow3DViewer] = useState(false);

  const modalities = [
    { name: "T1", color: "bg-blue-500" },
    { name: "T1ce", color: "bg-green-500" },
    { name: "T2", color: "bg-yellow-500" },
    { name: "FLAIR", color: "bg-purple-500" },
  ];

  return (
    <>
    <div className="w-80 h-full glass-panel p-4 space-y-6 custom-scrollbar overflow-y-auto">
      {/* File Upload Status */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Current Study
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            {modalities.map((modality) => (
              <Button
                key={modality.name}
                variant={selectedModality === modality.name ? "default" : "outline"}
                size="sm"
                className="justify-start gap-2"
                onClick={() => setSelectedModality(modality.name)}
              >
                <div className={`w-3 h-3 rounded-full ${modality.color}`} />
                {modality.name}
              </Button>
            ))}
          </div>
          
          <div className="pt-2">
            <Button className={`w-full gap-2 ${explainabilityMode ? 'bg-gradient-to-r from-yellow-500 to-orange-500' : 'bg-gradient-to-r from-primary to-accent'} hover:opacity-90 disabled:opacity-70`} onClick={onAnalyze} disabled={analyzing}>
              <Play className="h-4 w-4" />
              {analyzing ? `Analyzing… ${Math.floor(progress)}%` : (explainabilityMode ? "Analyze with Grad-CAM" : "Analyze Scans")}
            </Button>
            {analyzing && (
              <div className="mt-2 text-xs text-muted-foreground">This may take a minute depending on volume size.</div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Overlay Controls */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Layers className="h-5 w-5 text-accent" />
            Overlay Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Segmentation */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2">
                <Eye className="h-4 w-4 text-overlay-segmentation" />
                Segmented Regions
              </Label>
              <Switch
                checked={showSegmentation}
                onCheckedChange={setShowSegmentation}
              />
            </div>
            {showSegmentation && (
              <div className="space-y-2 pl-6">
                <Label className="text-sm text-muted-foreground">Opacity</Label>
                <Slider
                  value={segmentationOpacity}
                  onValueChange={setSegmentationOpacity}
                  max={1}
                  min={0}
                  step={0.1}
                  className="w-full"
                />
                <div className="flex flex-col gap-2 pt-1">
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={!!visibleLabels?.[1]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 1: e.target.checked })} />
                    <span className="flex items-center gap-2"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(25,118,210)' }} /> NCR/NET (1)</span>
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={!!visibleLabels?.[2]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 2: e.target.checked })} />
                    <span className="flex items-center gap-2"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(56,142,60)' }} /> Edema (2)</span>
                  </label>
                  <label className="flex items-center gap-2 text-sm">
                    <input type="checkbox" checked={!!visibleLabels?.[4]} onChange={(e)=> setVisibleLabels && setVisibleLabels({ ...(visibleLabels||{}), 4: e.target.checked })} />
                    <span className="flex items-center gap-2"><span className="inline-block w-3 h-3 rounded-sm" style={{ background:'rgb(211,47,47)' }} /> Enhancing (4)</span>
                  </label>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="outline" className="text-xs">
                    <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(25,118,210)' }} />
                    NCR/NET (1)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(56,142,60)' }} />
                    Edema (2)
                  </Badge>
                  <Badge variant="outline" className="text-xs">
                    <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(211,47,47)' }} />
                    Enhancing (4)
                  </Badge>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Explainability Mode */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-yellow-500" />
            AI Explainability
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="flex items-center gap-2">
                <Target className="h-4 w-4 text-yellow-500" />
                Explainability Mode
              </Label>
              <Switch
                checked={explainabilityMode}
                onCheckedChange={setExplainabilityMode}
              />
            </div>
            {explainabilityMode && (
              <div className="text-sm text-muted-foreground">
                This mode generates Grad-CAM heatmaps to show which regions the AI focuses on for tumor detection.
              </div>
            )}
          </div>

          {/* Grad-CAM Controls */}
          {explainabilityMode && gradcamHeatmaps && (
            <div className="space-y-3 pl-6">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Show Heatmaps</Label>
                <Switch
                  checked={showGradcam}
                  onCheckedChange={setShowGradcam}
                />
              </div>
              
              {showGradcam && (
                <>
                  <div className="space-y-2">
                    <Label className="text-sm text-muted-foreground">Heatmap Opacity</Label>
                    <Slider
                      value={gradcamOpacity}
                      onValueChange={setGradcamOpacity}
                      max={1}
                      min={0}
                      step={0.1}
                      className="w-full"
                    />
                  </div>
                  
                  <div className="flex flex-col gap-2 pt-1">
                    <div className="text-xs font-medium text-muted-foreground mb-1">ROI Heatmaps:</div>
                    {gradcamHeatmaps.necrotic && (
                      <Badge variant="outline" className="text-xs justify-start">
                        <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(255,193,7)' }} />
                        Necrotic Focus
                      </Badge>
                    )}
                    {gradcamHeatmaps.edema && (
                      <Badge variant="outline" className="text-xs justify-start">
                        <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(255,152,0)' }} />
                        Edema Focus
                      </Badge>
                    )}
                    {gradcamHeatmaps.enhancing && (
                      <Badge variant="outline" className="text-xs justify-start">
                        <div className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: 'rgb(244,67,54)' }} />
                        Enhancing Focus
                      </Badge>
                    )}
                  </div>
                </>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* 3D Visualization */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg flex items-center gap-2">
            <Box className="h-5 w-5 text-primary" />
            3D Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <Button 
            variant="outline" 
            className="w-full gap-2"
            onClick={() => setShow3DViewer(true)}
          >
            <Volume2 className="h-4 w-4" />
            3D Volume Render
          </Button>
        </CardContent>
      </Card>
    </div>
    
    {/* 3D Viewer Modal */}
    {show3DViewer && (
      <ThreeDViewer 
        onClose={() => setShow3DViewer(false)} 
        modality={selectedModality}
        modalityImages={modalityImages}
        segmentation={segmentation}
        labelNames={labelNames}
        visibleLabels={visibleLabels}
        setVisibleLabels={setVisibleLabels}
        gradcamHeatmaps={gradcamHeatmaps}
        showGradcam={showGradcam}
      />
    )}
    </>
  );
};