import { motion, AnimatePresence } from "framer-motion";
import { Badge } from "./ui/badge";
import { Sparkles, Cpu, Clock, Database, AlertTriangle, CheckCircle } from "lucide-react";
import type { PredictResponse } from "../types";

interface Props {
  result: PredictResponse;
}

const severityConfig = {
  "Severe Damage": {
    color: "text-danger",
    bg: "bg-red-50 border-red-200",
    icon: <AlertTriangle className="h-5 w-5 text-danger" />,
    badge: "bg-red-100 text-red-700 border-red-200",
  },
  "Mild Damage": {
    color: "text-yellow-500",
    bg: "bg-yellow-50 border-yellow-200",
    icon: <AlertTriangle className="h-5 w-5 text-yellow-500" />,
    badge: "bg-yellow-100 text-yellow-700 border-yellow-200",
  },
  "Little or No Damage": {
    color: "text-success",
    bg: "bg-green-50 border-green-200",
    icon: <CheckCircle className="h-5 w-5 text-success" />,
    badge: "bg-green-100 text-green-700 border-green-200",
  },
} as const;

export default function PredictionCard({ result }: Props) {
  const { prediction, model, processing_time_ms } = result;
  const cfg = severityConfig[prediction.severity as keyof typeof severityConfig] || severityConfig["Mild Damage"];
  const confidence = prediction.confidence;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="overflow-hidden rounded-xl border border-border bg-card shadow-lg"
    >
      <div className="bg-gradient-to-r from-success/5 to-primary/5 px-5 py-3.5 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            >
              <CheckCircle className="h-5 w-5 text-success" />
            </motion.div>
            <span className="text-sm font-semibold text-secondary">Prediction Complete</span>
          </div>
          <Badge className="bg-primary/10 text-primary border border-primary/20 text-xs flex items-center gap-1">
            <Sparkles className="h-3 w-3" />
            AI Powered
          </Badge>
        </div>
      </div>

      <div className="p-5 space-y-4">
        <div className={`rounded-lg border p-4 ${cfg.bg}`}>
          <div className="flex items-center gap-3">
            {cfg.icon}
            <div>
              <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Severity</p>
              <p className={`text-xl font-bold ${cfg.color}`}>{prediction.severity}</p>
            </div>
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</p>
            <span className="text-sm font-bold text-secondary">{confidence.toFixed(1)}%</span>
          </div>
          <div className="h-3 rounded-full bg-gray-100 overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${confidence.toFixed(1)}%` }}
              transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
              className="h-full rounded-full bg-gradient-to-r from-primary to-primary/70"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
              <Cpu className="h-3.5 w-3.5" />
              Model
            </div>
            <p className="text-sm font-semibold text-secondary">{model}</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
              <Clock className="h-3.5 w-3.5" />
              Inference
            </div>
            <p className="text-sm font-semibold text-secondary">{processing_time_ms.toFixed(0)} ms</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
              <Database className="h-3.5 w-3.5" />
              Dataset
            </div>
            <p className="text-sm font-semibold text-secondary">CrisisMMD v2</p>
          </div>
          <div className="rounded-lg bg-gray-50 p-3">
            <div className="flex items-center gap-1.5 text-xs text-gray-500 mb-1">
              <Sparkles className="h-3.5 w-3.5" />
              Status
            </div>
            <span className={`inline-flex items-center gap-1 text-sm font-semibold ${confidence > 70 ? "text-success" : "text-yellow-500"}`}>
              <span className="h-1.5 w-1.5 rounded-full bg-current" />
              {confidence > 70 ? "High Confidence" : "Medium Confidence"}
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
