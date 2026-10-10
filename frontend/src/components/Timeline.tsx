import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { CheckCircle2, Circle } from "lucide-react";

interface Stage {
  label: string;
  done: boolean;
}

const stages: Stage[] = [
  { label: "Dataset Analysis", done: true },
  { label: "Image Baseline", done: true },
  { label: "Text Baseline", done: true },
  { label: "Gated Fusion", done: true },
  { label: "Disaster Impact Index", done: true },
  { label: "Explainable AI", done: true },
  { label: "Decision Engine", done: true },
];

export default function Timeline() {
  return (
    <div className="w-full overflow-x-auto pb-4">
      <div className="flex min-w-max items-start gap-0">
        {stages.map((s, i) => (
          <div key={s.label} className="flex items-center">
            <div className="flex flex-col items-center">
              <motion.div
                initial={{ scale: 0 }}
                whileInView={{ scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors",
                  s.done
                    ? "border-success bg-success/10 text-success"
                    : "border-gray-200 bg-gray-50 text-gray-300"
                )}
              >
                {s.done ? <CheckCircle2 className="h-5 w-5" /> : <Circle className="h-5 w-5" />}
              </motion.div>
              <span
                className={cn(
                  "mt-2 text-xs font-medium whitespace-nowrap px-1",
                  s.done ? "text-secondary" : "text-gray-300"
                )}
              >
                {s.label}
              </span>
            </div>
            {i < stages.length - 1 && (
              <div
                className={cn(
                  "mx-1 mt-4 h-0.5 w-12 sm:w-16 md:w-20",
                  s.done ? "bg-success/40" : "bg-gray-100"
                )}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
