import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { checkHealth } from "@/services/api";
import { Wifi, WifiOff, Database, Cpu } from "lucide-react";

export default function SystemStatus() {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const res = await checkHealth();
        if (mounted) setHealthy(res.status === "healthy");
      } catch {
        if (mounted) setHealthy(false);
      }
    };
    poll();
    const interval = setInterval(poll, 15000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-border bg-card p-4 shadow-sm"
    >
      <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-400">
        System Status
      </h4>
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <Wifi className="h-3.5 w-3.5" />
            Backend
          </div>
          <span className={`flex items-center gap-1 text-xs font-medium ${healthy ? "text-success" : "text-danger"}`}>
            <span className={`h-2 w-2 rounded-full ${healthy === null ? "bg-yellow-400 animate-pulse" : healthy ? "bg-success" : "bg-danger"}`} />
            {healthy === null ? "Checking..." : healthy ? "Connected" : "Offline"}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <Cpu className="h-3.5 w-3.5" />
            Model
          </div>
          <span className="flex items-center gap-1 text-xs font-medium text-success">
            <span className="h-2 w-2 rounded-full bg-success" />
            Gated Fusion
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-gray-600">
            <Database className="h-3.5 w-3.5" />
            Dataset
          </div>
          <span className="text-xs font-medium text-gray-500">CrisisMMD v2</span>
        </div>
        <div className="mt-2 border-t border-border pt-2">
          <div className="flex items-center justify-between text-xs">
            <span className="text-gray-400">Version</span>
            <span className="font-medium text-secondary">v1.0 Final Release</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
