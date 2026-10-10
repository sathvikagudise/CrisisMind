import { Layout } from "@/components/Layout";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import FeatureCard from "@/components/FeatureCard";
import ComingSoonCard from "@/components/ComingSoonCard";
import AnimatedCounter from "@/components/AnimatedCounter";
import Timeline from "@/components/Timeline";
import { ArrowRight, Image, FileText, Brain, Zap, Truck, Clock, Database, GitMerge, Layers, CheckCircle } from "lucide-react";
import { Link } from "react-router-dom";

const features = [
  { icon: <Image className="h-5 w-5" />, title: "Image Analysis", description: "EfficientNet-B0 backbone extracts visual features from disaster images to identify structural damage patterns." },
  { icon: <FileText className="h-5 w-5" />, title: "Text Understanding", description: "DistilBERT processes social media text to understand contextual severity from crisis communications." },
  { icon: <Brain className="h-5 w-5" />, title: "Multimodal Fusion", description: "Gated Fusion model dynamically weights visual and textual signals for superior classification accuracy." },
  { icon: <Zap className="h-5 w-5" />, title: "Fast AI Prediction", description: "Optimized inference pipeline delivers severity classification in seconds on standard hardware." },
];

const techStack = [
  { name: "React", color: "text-sky-500", bg: "bg-sky-50 border-sky-200" },
  { name: "FastAPI", color: "text-emerald-600", bg: "bg-emerald-50 border-emerald-200" },
  { name: "PyTorch", color: "text-orange-600", bg: "bg-orange-50 border-orange-200" },
  { name: "EfficientNet-B0", color: "text-blue-600", bg: "bg-blue-50 border-blue-200" },
  { name: "DistilBERT", color: "text-violet-600", bg: "bg-violet-50 border-violet-200" },
  { name: "TailwindCSS", color: "text-cyan-500", bg: "bg-cyan-50 border-cyan-200" },
];

const completed = [
  { icon: <CheckCircle className="h-5 w-5" />, title: "Explainable AI", description: "Grad-CAM heatmaps and attention visualization for model interpretability.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Analytics Dashboard", description: "Comprehensive dashboard with charts, timeline, and system health.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Analysis Center", description: "Unified search, filter, and export across all completed analyses.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Batch Processing", description: "Asynchronous batch processing with CSV support and progress tracking.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Disaster Intelligence", description: "Real-time disaster event ingestion, analysis, and persistence.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "AI Assessment Reports", description: "Downloadable PDF and CSV reports with emergency recommendations.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Decision Engine", description: "Automated priority classification and emergency level assessment.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Emergency Recommendations", description: "Intelligent resource allocation based on predicted damage severity.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "Search & Filters", description: "Full-text search and multi-criteria filtering across all analyses.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "CSV Export", description: "Export filtered analysis data to CSV for external use.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "PDF Export", description: "Generate detailed PDF reports with Grad-CAM overlays.", done: true },
  { icon: <CheckCircle className="h-5 w-5" />, title: "REST API", description: "27-endpoint FastAPI backend with Swagger documentation.", done: true },
];

const stats = [
  { value: 18082, suffix: "", label: "Images Analyzed" },
  { value: 16059, suffix: "", label: "Crisis Tweets" },
  { value: 7, suffix: "", label: "Disaster Events" },
  { value: 4, suffix: "", label: "Deep Learning Models" },
];

export default function Home() {
  return (
    <Layout>
      {/* Hero */}
      <section className="relative overflow-hidden border-b border-border">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/[0.03] via-transparent to-accent/[0.03]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/[0.02] via-transparent to-transparent" />
        <div className="relative mx-auto max-w-7xl px-4 py-28 sm:px-6 sm:py-36 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="mx-auto max-w-3xl text-center"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className="mb-6 inline-flex items-center gap-1.5 rounded-full border border-border bg-white px-3 py-1 text-xs font-medium text-gray-600 shadow-sm"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              CrisisMind v3.0 — GatedFusionWide
            </motion.div>
            <h1 className="text-5xl font-extrabold tracking-tight text-secondary sm:text-6xl lg:text-7xl">
              Crisis<span className="text-primary">Mind</span>
            </h1>
            <p className="mt-4 text-lg font-medium text-gray-600 sm:text-xl">
              AI-Powered Multimodal Disaster Assessment and Emergency Response Intelligence
            </p>
            <p className="mt-3 text-base text-gray-400 max-w-xl mx-auto">
              Analyze disaster images and crisis-related reports using multimodal deep learning to assist emergency response planning.
            </p>
            <div className="mt-10 flex flex-col items-center justify-center gap-4 sm:flex-row">
              <Link to="/analysis">
                <Button size="lg">
                  Start Analysis
                  <ArrowRight className="h-4 w-4" />
                </Button>
              </Link>
              <a href="#features">
                <Button variant="outline" size="lg">
                  Learn More
                </Button>
              </a>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Research Statistics */}
      <section className="border-b border-border py-16">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 gap-6 sm:grid-cols-4">
            {stats.map((s, i) => (
              <motion.div
                key={s.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.1 }}
                className="rounded-xl border border-border bg-card p-6 text-center shadow-sm"
              >
                <AnimatedCounter value={s.value} suffix={s.suffix} label={s.label} />
              </motion.div>
            ))}
          </div>
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            className="mt-6 text-center"
          >
            <div className="inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/5 px-4 py-1.5 text-xs text-primary font-medium">
              <Layers className="h-3.5 w-3.5" />
              Best Model: Gated Fusion &mdash; Macro F1: 53.75%
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Core Capabilities
            </h2>
            <p className="mt-3 text-gray-500">
              Powered by cutting-edge multimodal deep learning
            </p>
          </motion.div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {features.map((f, i) => (
              <FeatureCard key={f.title} {...f} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Technology Stack
            </h2>
            <p className="mt-3 text-gray-500">
              Built with modern, production-grade frameworks
            </p>
          </motion.div>
          <div className="flex flex-wrap justify-center gap-4">
            {techStack.map((t, i) => (
              <motion.div
                key={t.name}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.05 }}
                className={`rounded-lg border ${t.bg} ${t.color} px-5 py-3 text-sm font-semibold shadow-sm`}
              >
                {t.name}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Progress Timeline */}
      <section className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Research Progress
            </h2>
            <p className="mt-3 text-gray-500">
              Development roadmap and completed milestones
            </p>
          </motion.div>
          <Timeline />
        </div>
      </section>

      {/* Research Highlights */}
      <section className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Research Highlights
            </h2>
            <p className="mt-3 text-gray-500">
              Key achievements and experimental results
            </p>
          </motion.div>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { icon: <Database className="h-4 w-4 text-primary" />, text: "18,082 disaster images analyzed across 7 event types" },
              { icon: <FileText className="h-4 w-4 text-primary" />, text: "16,059 crisis tweets processed for contextual understanding" },
              { icon: <GitMerge className="h-4 w-4 text-primary" />, text: "4 deep learning models evaluated and benchmarked" },
              { icon: <Brain className="h-4 w-4 text-primary" />, text: "Gated Fusion outperformed unimodal baselines by 10.4%" },
              { icon: <Zap className="h-4 w-4 text-primary" />, text: "Real-time multimodal inference on standard hardware" },
              { icon: <Layers className="h-4 w-4 text-primary" />, text: "Production-grade FastAPI backend with RESTful API" },
            ].map((h, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.08 }}
                className="flex items-start gap-3 rounded-lg border border-border bg-card p-4 shadow-sm"
              >
                <div className="mt-0.5 flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10">
                  {h.icon}
                </div>
                <p className="text-sm text-gray-600">{h.text}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Completed Modules */}
      <section className="border-b border-border py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Completed Modules
            </h2>
            <p className="mt-3 text-gray-500">
              Features implemented and available in this release
            </p>
          </motion.div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {completed.map((c, i) => (
              <ComingSoonCard key={c.title} {...c} index={i} />
            ))}
          </div>
        </div>
      </section>

      {/* Future Research */}
      <section className="py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-12 text-center"
          >
            <h2 className="text-2xl font-bold text-secondary sm:text-3xl">
              Future Research
            </h2>
            <p className="mt-3 text-gray-500">
              Potential extensions and research directions
            </p>
          </motion.div>
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[
              { title: "Live Satellite Imagery", desc: "Integration with real-time satellite feeds for continuous monitoring of disaster-affected regions." },
              { title: "Drone Image Assessment", desc: "Processing aerial/drone imagery for ground-level damage assessment at scale." },
              { title: "Multi-Language Reports", desc: "Disaster reports and analysis output in multiple languages for global deployment." },
              { title: "GIS Map Integration", desc: "Geospatial visualization of disaster impact zones on interactive maps." },
              { title: "Temporal Progression", desc: "Predicting how disaster damage evolves over time using sequential analysis." },
              { title: "Edge Deployment", desc: "Optimized models for on-device inference in low-connectivity disaster zones." },
            ].map((item, i) => (
              <ComingSoonCard
                key={item.title}
                icon={<Layers className="h-5 w-5" />}
                title={item.title}
                description={item.desc}
                index={i}
              />
            ))}
          </div>
        </div>
      </section>
    </Layout>
  );
}
