import { useState, useCallback } from "react";
import { Upload, File, CheckCircle, AlertCircle, Brain, FolderOpen } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface FileUploadZoneProps {
  onFilesUploaded: () => void;
  setModalityImages?: (images: { [modality: string]: any }) => void;
}

export const FileUploadZone = (props: FileUploadZoneProps) => {
  const { onFilesUploaded, setModalityImages } = props;
  const [modalityImages, setModalityImagesState] = useState<{ [modality: string]: any }>({});
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const fileInputRef = useCallback((node) => {}, []);
  const zipInputRef = useCallback((node) => {}, []);

  // Helper: Parse ZIP and extract files
  const handleZipFile = async (file: File) => {
    setIsUploading(true);
    setUploadProgress(0);
    try {
      const JSZip = (await import('jszip')).default;
      const zip = await JSZip.loadAsync(file);
      const files = Object.values(zip.files).filter(f => !f.dir);
      let extractedFiles: File[] = [];
      for (let i = 0; i < files.length; i++) {
        const zipEntry = files[i];
        const content = await zipEntry.async('blob');
        // Use Blob and manually set name property
        (content as any).name = zipEntry.name;
        extractedFiles.push(content as File);
        setUploadProgress(Math.round(((i + 1) / files.length) * 100));
      }
      handleFiles(extractedFiles);
    } catch (err) {
      alert('Failed to extract ZIP: ' + err);
    }
    setIsUploading(false);
  };

  // Helper: Parse DICOM/NIfTI files
  const handleFiles = async (files: File[]) => {
    setIsUploading(true);
    setUploadProgress(0);
    let modalityImages: { [modality: string]: any } = {};
    let uploaded: any[] = [];
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const type = getFileType(file.name);
      const modality = getModalityFromFilename(file.name);
      let status = 'pending';
      try {
        if (type === 'DICOM') {
          // DICOM parsing
          const dicomParser = (await import('dicom-parser')).default;
          const arrayBuffer = await file.arrayBuffer();
          const dataSet = dicomParser.parseDicom(new Uint8Array(arrayBuffer));
          modalityImages[modality] = dataSet;
          status = 'complete';
        } else if (type === 'NIfTI') {
          // NIfTI parsing
          const nifti = await import('nifti-reader-js');
          const arrayBuffer = await file.arrayBuffer();
          let niftiBuffer: ArrayBuffer = arrayBuffer;
          if (nifti.isCompressed(arrayBuffer)) {
            // nifti.decompress may return SharedArrayBuffer, convert to ArrayBuffer
            const decompressed = nifti.decompress(arrayBuffer);
            if (decompressed instanceof ArrayBuffer) {
              niftiBuffer = decompressed;
            } else {
              // Convert SharedArrayBuffer to ArrayBuffer
              niftiBuffer = new Uint8Array(decompressed).slice().buffer;
            }
          }
          if (nifti.isNIFTI(niftiBuffer)) {
            const niftiHeader = nifti.readHeader(niftiBuffer);
            const niftiImage = nifti.readImage(niftiHeader, niftiBuffer);
            modalityImages[modality] = { header: niftiHeader, image: niftiImage };
            status = 'complete';
          } else {
            status = 'error';
          }
        } else {
          status = 'error';
        }
      } catch (err) {
        status = 'error';
      }
      uploaded.push({ name: file.name, type, status });
      setUploadProgress(Math.round(((i + 1) / files.length) * 100));
    }
    setUploadedFiles(uploaded);
    setModalityImagesState(modalityImages);
    if (setModalityImages) setModalityImages(modalityImages);
    setIsUploading(false);
    if (onFilesUploaded) onFilesUploaded();
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };
  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    if (files.length === 1 && getFileType(files[0].name) === 'Archive') {
      handleZipFile(files[0]);
    } else {
      handleFiles(files);
    }
  };
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    if (files.length === 1 && getFileType(files[0].name) === 'Archive') {
      handleZipFile(files[0]);
    } else {
      handleFiles(files);
    }
  };

  const getFileType = (filename: string): string => {
    const ext = filename.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'dcm':
      case 'dicom':
        return 'DICOM';
      case 'nii':
      case 'gz':
        return 'NIfTI';
      case 'zip':
        return 'Archive';
      default:
        return 'Unknown';
    }
  };
  const getModalityFromFilename = (filename: string): string => {
    const name = filename.toLowerCase();

    // Order matters: handle specific variants before generic ones
    // T1 with contrast first
    if (name.includes('t1ce') || name.includes('t1c')) return 'T1ce';

    // FLAIR can appear as 'flair', 't2f', 't2_flair', 't2-flair', 't2flair'
    if (
      name.includes('flair') ||
      name.includes('t2f') ||
      name.includes('t2_flair') ||
      name.includes('t2-flair') ||
      name.includes('t2flair')
    ) {
      return 'FLAIR';
    }

    // T2 (avoid catching T2-FLAIR which is handled above)
    if ((name.includes('t2w') || name.includes('t2')) && !name.includes('t2f')) return 'T2';

    // T1 (avoid catching T1ce above)
    if (name.includes('t1w') || name.includes('t1') || name.includes('t1n')) return 'T1';

    return 'Unknown';
  };

  return (
    <div className="h-full flex items-center justify-center overflow-auto">
      <Card className="w-full max-w-2xl">
        <CardContent className="p-8">
          <div className="text-center space-y-6">
            <div className="w-16 h-16 bg-gradient-to-br from-primary to-accent rounded-full flex items-center justify-center mx-auto">
              <Brain className="h-8 w-8 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-foreground mb-2">Upload MRI Scans</h2>
              <p className="text-muted-foreground">Upload DICOM files, NIfTI volumes, or a zip archive containing your brain MRI data</p>
            </div>
            {/* Upload Zone */}
            <div
              className={`upload-zone relative rounded-lg p-8 text-center cursor-pointer ${isDragOver ? 'drag-over' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input')?.click()}
            >
              <input
                id="file-input"
                type="file"
                multiple
                accept=".dcm,.dicom,.nii,.nii.gz,.zip"
                className="hidden"
                onChange={handleFileSelect}
                ref={fileInputRef}
              />
              <div className="space-y-4">
                <Upload className="h-12 w-12 text-muted-foreground mx-auto" />
                <div>
                  <p className="text-lg font-medium">Drop your files here or click to select</p>
                  <p className="text-sm text-muted-foreground mt-2">Supports: DICOM (.dcm), NIfTI (.nii, .nii.gz), ZIP archives</p>
                </div>
              </div>
              {isDragOver && (
                <div className="absolute inset-0 bg-primary/10 border-2 border-primary border-dashed rounded-lg flex items-center justify-center">
                  <p className="text-primary font-medium">Drop files here</p>
                </div>
              )}
            </div>
            {/* Expected File Types */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['T1', 'T1ce', 'T2', 'FLAIR'].map((modality) => (
                <div key={modality} className="p-3 border border-border rounded-lg">
                  <div className="text-sm font-medium">{modality}</div>
                  <div className="text-xs text-muted-foreground">{modality === 'T1ce' ? 'Contrast Enhanced' : 'MRI Sequence'}</div>
                </div>
              ))}
            </div>
            {/* Upload Progress */}
            {isUploading && (
              <div className="space-y-4">
                <Progress value={uploadProgress} className="w-full" />
                <p className="text-sm text-muted-foreground">Uploading files... {uploadProgress}%</p>
              </div>
            )}
            {/* Uploaded Files */}
            {uploadedFiles.length > 0 && (
              <div className="space-y-3">
                <h3 className="font-medium text-left">Uploaded Files</h3>
                <div className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border border-border rounded-lg">
                      <div className="flex items-center gap-3">
                        <File className="h-4 w-4 text-muted-foreground" />
                        <div className="text-left">
                          <div className="text-sm font-medium truncate">{file.name}</div>
                          <div className="text-xs text-muted-foreground">{file.type} • {getModalityFromFilename(file.name)}</div>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={file.status === 'complete' ? 'default' : 'secondary'}>{getModalityFromFilename(file.name)}</Badge>
                        {file.status === 'complete' ? (
                          <CheckCircle className="h-4 w-4 text-success" />
                        ) : file.status === 'error' ? (
                          <AlertCircle className="h-4 w-4 text-destructive" />
                        ) : (
                          <div className="w-4 h-4 border-2 border-primary border-t-transparent rounded-full animate-spin" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
            {/* Alternative Upload Options */}
            <input
              id="zip-input"
              type="file"
              accept=".zip"
              className="hidden"
              onChange={handleFileSelect}
              ref={zipInputRef}
            />
            <div className="flex gap-4 justify-center">
              <Button variant="outline" className="gap-2" onClick={() => document.getElementById('file-input')?.click()}>
                <FolderOpen className="h-4 w-4" />
                Browse Files
              </Button>
              <Button variant="outline" className="gap-2" onClick={() => document.getElementById('zip-input')?.click()}>
                <Upload className="h-4 w-4" />
                Upload ZIP
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}