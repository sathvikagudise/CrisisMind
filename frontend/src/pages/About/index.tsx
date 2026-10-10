import { Layout } from "@/components/Layout";
import { motion } from "framer-motion";
import { FlaskConical, BookOpen, Database, GitMerge, Brain, Layers, BarChart3, Target, Lightbulb, Image, FileText, Cpu } from "lucide-react";

const cards = [
  {
    icon: <BookOpen className="h-5 w-5" />,
    title: "Project Overview",
    content: "CrisisMind is a multimodal deep learning framework that classifies disaster severity from social media images and text. It combines visual features from EfficientNet-B0 with textual context from DistilBERT using a novel Gated Fusion architecture, achieving state-of-the-art results on the CrisisMMD v2.0 dataset.",
  },
  {
    icon: <Target className="h-5 w-5" />,
    title: "Problem Statement",
    content: "During natural disasters, emergency responders face critical information overload from social media. Manual assessment of thousands of posts is slow, error-prone, and delays response time. An automated multimodal AI system that simultaneously understands images and text is urgently needed to triage damage severity in real-time and support emergency decision-making.",
  },
  {
    icon: <Database className="h-5 w-5" />,
    title: "Dataset",
    content: "CrisisMMD v2.0 is a multimodal crisis dataset containing 18,082 image-text pairs collected from Twitter across 7 disaster events including earthquakes, hurricanes, floods, wildfires, and cyclones. Of these, 3,526 samples are human-annotated with damage severity labels across three classes: Little or No Damage, Moderate Damage, and Severe Damage.",
    badge: "CrisisMMD v2.0",
  },
  {
    icon: <GitMerge className="h-5 w-5" />,
    title: "Methodology",
    items: [
      { label: "Image Baseline", value: "EfficientNet-B0 (frozen backbone, 3,843 trainable params)", icon: <Image className="h-4 w-4" /> },
      { label: "Text Baseline", value: "DistilBERT (fine-tuned, max 64 tokens)", icon: <FileText className="h-4 w-4" /> },
      { label: "Concat Fusion", value: "Concatenated 1280-d visual + 768-d text features to MLP", icon: <Layers className="h-4 w-4" /> },
      { label: "Gated Fusion", value: "Learnable per-sample modality weighting (Best: Macro F1 53.75%)", icon: <Brain className="h-4 w-4" /> },
    ],
  },
  {
    icon: <Lightbulb className="h-5 w-5" />,
    title: "Research Contributions",
    items: [
      "Comprehensive benchmark of unimodal vs. multimodal approaches on CrisisMMD",
      "Novel Gated Fusion architecture with learnable per-sample modality gating",
      "Pre-extraction + caching strategy for CPU-efficient multimodal training",
      "Production-grade FastAPI backend with multimodal inference pipeline",
      "Full reproducibility package: checkpoints, metrics, and codebase",
    ],
  },
  {
    icon: <BarChart3 className="h-5 w-5" />,
    title: "Implemented Features",
    items: [
      "Disaster Impact Index — continuous 0-100 severity scoring",
      "Explainable AI — Grad-CAM heatmaps and attention visualization",
      "Emergency Decision Engine — risk-based resource recommendation",
      "Batch Processing — asynchronous CSV and single-image analysis",
      "Analysis Center — unified search, filter, and export",
      "REST API — 27-endpoint FastAPI backend with Swagger docs",
    ],
  },
  {
    icon: <Layers className="h-5 w-5" />,
    title: "Future Research",
    items: [
      "Live satellite imagery for real-time continuous monitoring",
      "Drone image processing for ground-level damage assessment",
      "Multi-language disaster reports for global deployment",
      "GIS map integration for geospatial impact visualization",
    ],
  },
];

const techIcons: Record<string, React.ReactNode> = {
  React: <Cpu className="h-4 w-4" />,
  FastAPI: <Cpu className="h-4 w-4" />,
  PyTorch: <Cpu className="h-4 w-4" />,
  "EfficientNet-B0": <Image className="h-4 w-4" />,
  DistilBERT: <FileText className="h-4 w-4" />,
  TailwindCSS: <Cpu className="h-4 w-4" />,
};

const techStack = [
  { name: "React", color: "text-sky-500", bg: "bg-sky-50 border-sky-200" },
  { name: "FastAPI", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200" },
  { name: "PyTorch", color: "text-orange-600", bg: "bg-orange-50 border-orange-200" },
  { name: "EfficientNet-B0", color: "text-blue-600", bg: "bg-blue-50 border-blue-200" },
  { name: "DistilBERT", color: "text-violet-600", bg: "bg-violet-50 border-violet-200" },
  { name: "TailwindCSS", color: "text-cyan-500", bg: "bg-cyan-50 border-cyan-200" },
];

export default function About() {
  return (
    <Layout>
      <div className="mx-auto max-w-5xl px-4 py-12 sm:px-6 lg:px-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-12 text-center">
          <div className="mb-4 inline-flex items-center justify-center h-14 w-14 rounded-2xl bg-primary/10">
            <FlaskConical className="h-7 w-7 text-primary" />
          </div>
          <h1 className="text-3xl font-bold text-secondary sm:text-4xl">
            About <span className="text-primary">CrisisMind</span>
          </h1>
          <p className="mt-4 text-lg text-gray-500">
            Research project overview, methodology, and technical details
          </p>
        </motion.div>

        <div className="grid gap-6 md:grid-cols-2">
          {cards.map((card, idx) => (
            <motion.div
              key={card.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.08 }}
              className={`rounded-xl border border-border bg-card p-6 shadow-sm ${
                card.title === "Project Overview" || card.title === "Methodology" || card.title === "Research Contributions"
                  ? "md:col-span-2"
                  : ""
              }`}
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                  {card.icon}
                </div>
                <h2 className="text-lg font-bold text-secondary">{card.title}</h2>
                {card.badge && (
                  <span className="ml-auto rounded-full bg-primary/10 px-2.5 py-0.5 text-xs font-medium text-primary">
                    {card.badge}
                  </span>
                )}
              </div>

              {card.content && <p className="text-sm leading-relaxed text-gray-600">{card.content}</p>}

              {card.items && card.title === "Methodology" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  {(card.items as { icon: React.ReactNode; label: string; value: string }[]).map((item) => (
                    <div key={item.label} className="rounded-lg border border-border bg-gray-50 p-3">
                      <div className="flex items-center gap-2 text-sm font-semibold text-secondary mb-1">
                        {item.icon}
                        {item.label}
                      </div>
                      <p className="text-xs text-gray-500">{item.value}</p>
                    </div>
                  ))}
                </div>
              )}

              {card.items && card.title === "Research Contributions" && (
                <ul className="space-y-2">
                  {(card.items as string[]).map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-gray-600">
                      <span className="mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary/60" />
                      {item}
                    </li>
                  ))}
                </ul>
              )}

              {card.items && card.title === "Implemented Features" && (
                <div className="grid gap-3 sm:grid-cols-3">
                  {(card.items as string[]).map((item, i) => (
                    <div key={i} className="rounded-lg border border-success/30 bg-success/5 p-3 text-center">
                      <p className="text-xs text-gray-600">{item}</p>
                    </div>
                  ))}
                </div>
              )}

              {card.items && card.title === "Future Research" && (
                <div className="grid gap-3 sm:grid-cols-2">
                  {(card.items as string[]).map((item, i) => (
                    <div key={i} className="rounded-lg border border-dashed border-gray-200 bg-gray-50/50 p-3 text-center">
                      <p className="text-xs text-gray-400">{item}</p>
                    </div>
                  ))}
                </div>
              )}
            </motion.div>
          ))}
        </div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="mt-8 rounded-xl border border-border bg-card p-6 shadow-sm"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
              <Cpu className="h-5 w-5" />
            </div>
            <h2 className="text-lg font-bold text-secondary">Technology Stack</h2>
          </div>
          <div className="flex flex-wrap gap-3">
            {techStack.map((t) => (
              <div key={t.name} className={`rounded-lg border ${t.bg} ${t.color} px-4 py-2 text-sm font-semibold shadow-sm`}>
                {t.name}
              </div>
            ))}
          </div>
        </motion.div>

        {/* GitHub link */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="mt-8 text-center"
        >
          <a
            href="https://github.com/sathvikrio/CrisisMind"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 text-sm text-gray-500 hover:text-primary transition-colors"
          >
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
            View on GitHub
          </a>
        </motion.div>
      </div>
    </Layout>
  );
}
