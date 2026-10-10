import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface Props {
  icon: ReactNode;
  title: string;
  description: string;
  className?: string;
  index?: number;
}

export default function FeatureCard({
  icon,
  title,
  description,
  className,
  index = 0,
}: Props) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-50px" }}
      transition={{ duration: 0.4, delay: index * 0.1 }}
      className={cn(
        "group rounded-xl border border-border bg-card p-6 shadow-sm transition-all duration-300 hover:shadow-md hover:-translate-y-0.5",
        className
      )}
    >
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
        {icon}
      </div>
      <h3 className="mb-2 text-base font-semibold">{title}</h3>
      <p className="text-sm leading-relaxed text-gray-500">{description}</p>
    </motion.div>
  );
}
