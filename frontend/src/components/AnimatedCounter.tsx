import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";

interface Props {
  value: number;
  suffix?: string;
  duration?: number;
  label?: string;
}

export default function AnimatedCounter({ value, suffix = "", duration = 2, label }: Props) {
  const [count, setCount] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    let start = 0;
    const increment = value / (duration * 60);
    const timer = setInterval(() => {
      start += increment;
      if (start >= value) {
        setCount(value);
        clearInterval(timer);
      } else {
        setCount(Math.floor(start));
      }
    }, 16);
    return () => clearInterval(timer);
  }, [inView, value, duration]);

  return (
    <div ref={ref} className="text-center">
      <motion.span
        initial={{ opacity: 0, y: 10 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        className="text-3xl font-bold text-secondary sm:text-4xl"
      >
        {count.toLocaleString()}{suffix}
      </motion.span>
      {label && <p className="mt-1 text-xs text-gray-400">{label}</p>}
    </div>
  );
}
