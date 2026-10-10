import { motion } from "framer-motion";
import { Image, Download, Brain, Info } from "lucide-react";
import type { GradCAMResponse } from "../types";

interface Props {
  explanation: GradCAMResponse;
  hasText: boolean;
}

const SERVER_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function imageUrl(path: string): string {
  if (path.startsWith("http")) return path;
  return `${SERVER_URL}${path}`;
}

export default function ExplanationCard({ explanation, hasText }: Props) {
  const { images, prediction, explanation: meta } = explanation;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="overflow-hidden rounded-xl border border-border bg-card shadow-lg"
    >
      <div className="bg-gradient-to-r from-blue-50 to-primary/5 px-5 py-3.5 border-b border-border">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" />
          <span className="text-sm font-semibold text-secondary">AI Explanation (Grad-CAM)</span>
        </div>
      </div>

      <div className="p-5 space-y-4">
        {hasText && (
          <div className="rounded-lg bg-blue-50 border border-blue-200 px-4 py-3">
            <div className="flex items-start gap-2">
              <Info className="h-4 w-4 text-blue-500 mt-0.5 shrink-0" />
              <p className="text-xs text-blue-700 leading-relaxed">
                The visualization explains the image contribution. The final prediction also considered the accompanying text.
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* Original */}
          <div className="rounded-lg border border-border overflow-hidden bg-gray-50">
            <div className="bg-gray-100 px-3 py-1.5 border-b border-border flex items-center gap-1.5">
              <Image className="h-3.5 w-3.5 text-gray-500" />
              <span className="text-xs font-medium text-gray-600">Original</span>
            </div>
            <a href={imageUrl(images.original)} target="_blank" rel="noopener noreferrer">
              <img
                src={imageUrl(images.original)}
                alt="Original"
                className="w-full h-40 object-contain cursor-pointer hover:opacity-90 transition-opacity"
              />
            </a>
          </div>

          {/* Heatmap */}
          <div className="rounded-lg border border-border overflow-hidden bg-gray-50">
            <div className="bg-gray-100 px-3 py-1.5 border-b border-border flex items-center gap-1.5">
              <Image className="h-3.5 w-3.5 text-orange-500" />
              <span className="text-xs font-medium text-gray-600">Heatmap</span>
            </div>
            <a href={imageUrl(images.heatmap)} target="_blank" rel="noopener noreferrer">
              <img
                src={imageUrl(images.heatmap)}
                alt="Heatmap"
                className="w-full h-40 object-contain cursor-pointer hover:opacity-90 transition-opacity"
              />
            </a>
          </div>

          {/* Overlay */}
          <div className="rounded-lg border border-border overflow-hidden bg-gray-50">
            <div className="bg-gray-100 px-3 py-1.5 border-b border-border flex items-center gap-1.5">
              <Image className="h-3.5 w-3.5 text-red-500" />
              <span className="text-xs font-medium text-gray-600">Overlay</span>
            </div>
            <a href={imageUrl(images.overlay)} target="_blank" rel="noopener noreferrer">
              <img
                src={imageUrl(images.overlay)}
                alt="Overlay"
                className="w-full h-40 object-contain cursor-pointer hover:opacity-90 transition-opacity"
              />
            </a>
          </div>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-xs text-gray-400">
            Technique: <span className="font-medium text-gray-600">{meta.technique}</span>
            &nbsp;&middot;&nbsp;
            Layer: <span className="font-medium text-gray-600">{meta.target_layer}</span>
            &nbsp;&middot;&nbsp;
            Predicted: <span className="font-medium text-gray-600">{prediction.severity} ({prediction.confidence.toFixed(1)}%)</span>
          </div>
          <a
            href={imageUrl(images.overlay)}
            download="gradcam_overlay.png"
            className="inline-flex items-center gap-1 rounded-lg border border-primary/30 bg-primary/5 px-3 py-1.5 text-xs font-medium text-primary transition-colors hover:bg-primary/10"
          >
            <Download className="h-3.5 w-3.5" />
            Overlay
          </a>
        </div>

        <div className="rounded-lg bg-gray-50 border border-border px-4 py-3">
          <p className="text-xs text-gray-500 leading-relaxed">
            The highlighted regions contributed most to the model&apos;s prediction. Bright red/yellow areas indicate higher influence on the predicted severity.
          </p>
        </div>
      </div>
    </motion.div>
  );
}
