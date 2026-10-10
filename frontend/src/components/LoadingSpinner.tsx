import { motion } from "framer-motion";
import { useEffect, useState } from "react";

const messages = [
  "Uploading image...",
  "Extracting visual features...",
  "Understanding disaster report...",
  "Running multimodal fusion...",
  "Generating prediction...",
];

export default function LoadingSpinner({ text = "Analyzing disaster..." }) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (text !== "Analyzing disaster...") return;
    const timer = setInterval(() => {
      setIndex((i) => (i + 1) % messages.length);
    }, 1800);
    return () => clearInterval(timer);
  }, [text]);

  const display = text === "Analyzing disaster..." ? messages[index] : text;

  return (
    <div className="flex flex-col items-center justify-center gap-4 py-10">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
        className="h-10 w-10 rounded-full border-4 border-primary/20 border-t-primary"
      />
      <motion.p
        key={display}
        initial={{ opacity: 0, y: 5 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -5 }}
        className="text-sm font-medium text-gray-500"
      >
        {display}
      </motion.p>
    </div>
  );
}
