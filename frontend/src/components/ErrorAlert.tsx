import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle, AlertTriangle, X } from "lucide-react";
import { Button } from "./ui/button";

interface Props {
  message: string;
  type?: "error" | "warning";
  onRetry?: () => void;
  onDismiss?: () => void;
  visible: boolean;
}

export default function ErrorAlert({
  message,
  type = "error",
  onRetry,
  onDismiss,
  visible,
}: Props) {
  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className={`rounded-lg border p-4 ${
            type === "error"
              ? "border-red-200 bg-red-50"
              : "border-yellow-200 bg-yellow-50"
          }`}
        >
          <div className="flex items-start gap-3">
            {type === "error" ? (
              <AlertCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-danger" />
            ) : (
              <AlertTriangle className="mt-0.5 h-5 w-5 flex-shrink-0 text-yellow-500" />
            )}
            <div className="flex-1">
              <p
                className={`text-sm font-medium ${
                  type === "error" ? "text-red-800" : "text-yellow-800"
                }`}
              >
                {message}
              </p>
              {onRetry && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onRetry}
                  className="mt-2 h-8 text-xs"
                >
                  Retry
                </Button>
              )}
            </div>
            {onDismiss && (
              <button
                onClick={onDismiss}
                className="rounded p-0.5 text-gray-400 hover:text-gray-600"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
