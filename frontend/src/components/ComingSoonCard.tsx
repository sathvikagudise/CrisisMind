import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { Lock, CheckCircle2 } from "lucide-react";

interface Props {
  title: string;
  description: string;
  icon: React.ReactNode;
  className?: string;
  index?: number;
  done?: boolean;
}

export default function ComingSoonCard({
  title,
  description,
  icon,
  className,
  index = 0,
  done,
}: Props) {
  if (done) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-50px" }}
        transition={{ duration: 0.4, delay: index * 0.1 }}
        className={cn(
          "relative overflow-hidden rounded-xl border border-green-200 bg-green-50 p-6 shadow-sm",
          className
        )}
      >
        <div className="absolute right-3 top-3 rounded-full bg-green-100 px-2 py-0.5 text-[10px] font-semibold text-green-600 flex items-center gap-1">
          <CheckCircle2 className="h-3 w-3" />
          Completed
        </div>
        <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-green-100 text-green-600">
          {icon}
        </div>
        <h3 className="mb-2 text-base font-semibold text-green-800">{title}</h3>
        <p className="text-sm leading-relaxed text-green-700">{description}</p>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className={cn(
        "relative overflow-hidden rounded-xl border border-border bg-card p-6 shadow-sm opacity-60 grayscale",
        className
      )}
    >
      <div className="absolute right-3 top-3 rounded-full bg-gray-100 px-2 py-0.5 text-[10px] font-semibold text-gray-400 flex items-center gap-1">
        <Lock className="h-3 w-3" />
        Coming Soon
      </div>
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-gray-100 text-gray-400">
        {icon}
      </div>
      <h3 className="mb-2 text-base font-semibold text-gray-400">{title}</h3>
      <p className="text-sm leading-relaxed text-gray-400">{description}</p>
    </motion.div>
  );
}
