import { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, X, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface Props {
  onFileSelect: (file: File | null) => void;
  file: File | null;
  disabled?: boolean;
}

function getResolution(file: File): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => resolve(`${img.width} × ${img.height}`);
    img.onerror = () => resolve("Unknown");
    img.src = URL.createObjectURL(file);
  });
}

export default function UploadCard({ onFileSelect, file, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [resolution, setResolution] = useState<string>("");

  const handleFile = useCallback(
    (f: File) => {
      if (!f.type.startsWith("image/")) return;
      if (f.size > 10 * 1024 * 1024) return;
      onFileSelect(f);
      const reader = new FileReader();
      reader.onload = () => setPreview(reader.result as string);
      reader.readAsDataURL(f);
      getResolution(f).then(setResolution);
    },
    [onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  const clear = () => {
    onFileSelect(null);
    setPreview(null);
    setResolution("");
  };

  return (
    <AnimatePresence mode="wait">
      {preview ? (
        <motion.div
          key="preview"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
        >
          <div className="relative overflow-hidden rounded-xl border border-border bg-card">
            <img src={preview} alt="Preview" className="h-56 w-full object-cover" />
            <button
              onClick={clear}
              disabled={disabled}
              className="absolute right-2 top-2 rounded-full bg-black/60 p-1 text-white hover:bg-black/80 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="mt-3 rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-secondary truncate max-w-[200px]">{file?.name}</span>
              <span className="flex items-center gap-1 text-xs font-medium text-success">
                <CheckCircle className="h-3.5 w-3.5" />
                Image Ready
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-xs text-gray-400">
              <span>{(file!.size / 1024 / 1024).toFixed(1)} MB</span>
              <span>{resolution}</span>
              <span className="text-right">{file?.name.split(".").pop()?.toUpperCase()}</span>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          key="upload"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={cn(
            "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 transition-colors",
            dragOver
              ? "border-primary bg-primary/5"
              : "border-border bg-card hover:border-primary/50 hover:bg-gray-50"
          )}
          onClick={() => !disabled && document.getElementById("image-input")?.click()}
        >
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10 text-primary">
            <Upload className="h-5 w-5" />
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-secondary">
              Drag & drop or <span className="text-primary">browse</span>
            </p>
            <p className="mt-1 text-xs text-gray-400">
              JPG, JPEG, PNG &bull; up to 10 MB
            </p>
          </div>
          <input
            id="image-input"
            type="file"
            accept="image/jpeg,image/png,image/jpg"
            className="hidden"
            onChange={handleChange}
            disabled={disabled}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
