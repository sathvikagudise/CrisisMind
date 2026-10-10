import { useState } from "react";
import { motion } from "framer-motion";
import { Send, AlertCircle, ArrowRight, FileText, Brain, EyeOff } from "lucide-react";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import UploadCard from "@/components/UploadCard";
import PredictionCard from "@/components/PredictionCard";
import ExplanationCard from "@/components/ExplanationCard";
import LoadingSpinner from "@/components/LoadingSpinner";
import SystemStatus from "@/components/SystemStatus";
import ErrorAlert from "@/components/ErrorAlert";
import { predict, getReportManualUrl, generateGradCAM } from "@/services/api";
import type { PredictResponse, GradCAMResponse } from "@/types";

const steps = [
  { num: 1, label: "Upload Disaster Image", done: false },
  { num: 2, label: "Describe the Disaster", done: false },
  { num: 3, label: "Run AI Analysis", done: false },
  { num: 4, label: "Prediction Result", done: false },
];

export default function Analysis() {
  const [image, setImage] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<GradCAMResponse | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);

  const canSubmit = image !== null && text.trim().length > 0 && !loading;
  const hasImage = image !== null;
  const isTextOnly = !image && text.trim().length > 0;
  const activeStep = result ? 4 : loading ? 3 : image && text.trim() ? 3 : image ? 2 : 1;
  const activeSteps = steps.map((s, i) => ({ ...s, done: i < activeStep - 1 || (i === 3 && !!result), active: i === activeStep - 1 }));

  const handleSubmit = async () => {
    if (!image || !text.trim()) return;
    setLoading(true);
    setResult(null);
    setError(null);
    try {
      const data = await predict(image, text.trim());
      setResult(data);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Prediction failed. Please try again.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = () => {
    setError(null);
    handleSubmit();
  };

  const handleExplanation = async () => {
    if (!image || !result?.analysis_id) return;
    setExplanationLoading(true);
    setExplanationError(null);
    setExplanation(null);
    try {
      const data = await generateGradCAM(image, text.trim(), result.analysis_id);
      setExplanation(data);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Grad-CAM generation failed.";
      setExplanationError(msg);
    } finally {
      setExplanationLoading(false);
    }
  };

  return (
    <Layout>
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-2xl font-bold text-secondary sm:text-3xl">Disaster Analysis</h1>
          <p className="mt-2 text-gray-500">Upload a disaster image and enter crisis-related text for AI-powered severity classification.</p>
        </motion.div>

        {/* Step Indicator */}
        <div className="mb-8 flex items-center gap-0 overflow-x-auto">
          {activeSteps.map((s, i) => (
            <div key={s.num} className="flex items-center">
              <div className={`flex items-center gap-2 whitespace-nowrap rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                s.active ? "bg-primary text-white shadow-sm" : s.done ? "bg-primary/10 text-primary" : "bg-gray-50 text-gray-400"
              }`}>
                <span className="flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold border border-current">
                  {s.done ? "\u2713" : s.num}
                </span>
                {s.label}
              </div>
              {i < steps.length - 1 && <div className={`mx-2 h-px w-8 ${s.done || s.active ? "bg-primary/30" : "bg-gray-200"}`} />}
            </div>
          ))}
        </div>

        <div className="grid gap-8 lg:grid-cols-5">
          {/* Left Panel - Input */}
          <div className="space-y-6 lg:col-span-3">
            {/* Step 1 */}
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">1</span>
                <h3 className="text-sm font-semibold text-secondary">Upload Disaster Image</h3>
                {image && <span className="text-xs text-success font-medium">Done</span>}
              </div>
              <UploadCard onFileSelect={setImage} file={image} disabled={loading} />
            </motion.div>

            {/* Step 2 */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">2</span>
                <h3 className="text-sm font-semibold text-secondary">Describe the Disaster</h3>
                {text.trim() && <span className="text-xs text-success font-medium">Done</span>}
              </div>
              <div className="rounded-xl border border-border bg-card overflow-hidden">
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value.slice(0, 500))}
                  disabled={loading}
                  placeholder='Example: "Several buildings have collapsed, roads are blocked by debris, and emergency rescue teams are evacuating residents."'
                  className="w-full resize-none border-0 bg-transparent p-4 text-sm text-secondary placeholder:text-gray-400 focus:outline-none disabled:opacity-50"
                  rows={4}
                />
                <div className="flex items-center justify-between border-t border-border px-4 py-2">
                  <span className="text-xs text-gray-400">Describe the situation shown in the image</span>
                  <span className="text-xs text-gray-400">{text.length}/500</span>
                </div>
              </div>
            </motion.div>

            {/* Step 3 - Analyze */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
            >
              <div className="flex items-center gap-2 mb-3">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">3</span>
                <h3 className="text-sm font-semibold text-secondary">Run AI Analysis</h3>
              </div>
              <Button
                size="lg"
                onClick={handleSubmit}
                disabled={!canSubmit}
                className="w-full h-12 text-base"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                    Analyzing...
                  </span>
                ) : (
                  <>
                    <Send className="h-4 w-4" />
                    Analyze Disaster
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </Button>
            </motion.div>
          </div>

          {/* Right Panel - Results */}
          <div className="lg:col-span-2">
            <div className="sticky top-24 space-y-6">
              {/* Step 4 - Result */}
              <div className="flex items-center gap-2 mb-1">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">4</span>
                <h3 className="text-sm font-semibold text-secondary">Prediction Result</h3>
              </div>

              <ErrorAlert
                message={error || ""}
                type="error"
                visible={!!error}
                onRetry={handleRetry}
                onDismiss={() => setError(null)}
              />

              {loading && <LoadingSpinner />}

              {result && !loading && !error && (
                <>
                  <PredictionCard result={result} />

                  {/* AI Explanation */}
                  <div className="space-y-3">
                    {hasImage ? (
                      <Button
                        variant="outline"
                        size="lg"
                        onClick={handleExplanation}
                        disabled={explanationLoading}
                        className="w-full h-12 text-base border-primary/30 text-primary hover:bg-primary/5"
                      >
                        {explanationLoading ? (
                          <span className="flex items-center gap-2">
                            <span className="h-4 w-4 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
                            Generating Explanation...
                          </span>
                        ) : (
                          <>
                            <Brain className="h-4 w-4" />
                            AI Explanation (Grad-CAM)
                          </>
                        )}
                      </Button>
                    ) : (
                      <div className="flex items-center gap-2 rounded-lg border border-gray-200 bg-gray-50 px-4 py-3">
                        <EyeOff className="h-4 w-4 text-gray-400" />
                        <p className="text-xs text-gray-400">
                          AI Explanation is available only for analyses containing an image.
                        </p>
                      </div>
                    )}
                  </div>

                  {explanationError && (
                    <ErrorAlert
                      message={explanationError}
                      type="error"
                      visible={true}
                      onRetry={handleExplanation}
                      onDismiss={() => setExplanationError(null)}
                    />
                  )}

                  {explanation && (
                    <ExplanationCard
                      explanation={explanation}
                      hasText={text.trim().length > 0}
                    />
                  )}

                  <div className="pt-1">
                    <a
                      href={getReportManualUrl(result.analysis_id || "")}
                      download
                      className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-primary/30 bg-primary/5 px-4 py-2.5 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
                    >
                      <FileText className="h-4 w-4" />
                      Download PDF Report
                    </a>
                  </div>
                </>
              )}

              {!result && !loading && !error && (
                <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-card p-8 text-center">
                  <AlertCircle className="h-8 w-8 text-gray-300 mb-3" />
                  <p className="text-sm text-gray-400">
                    Complete steps 1-3, then click <strong>Analyze</strong> to see the AI prediction.
                  </p>
                </div>
              )}

              {/* System Status */}
              <SystemStatus />
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}
