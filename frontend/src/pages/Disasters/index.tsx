import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RefreshCw,
  AlertTriangle,
  MapPin,
  Globe,
  Clock,
  Cpu,
  Activity,
  CheckCircle,
  XCircle,
  BarChart3,
  List,
  Flame,
  Droplets,
  Wind,
  CloudLightning,
  Mountain,
  AlertCircle,
  Satellite,
  FileText,
} from "lucide-react";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import {
  fetchDisasterEvents,
  analyzeDisasterEvent,
  getReportLiveUrl,
} from "@/services/api";
import type {
  DisasterEventSummary,
  DisasterAnalysisResponse,
} from "@/types";

// ── Helpers ──────────────────────────────────────────────────────────

const typeIcon: Record<string, React.ReactNode> = {
  wildfire: <Flame className="h-5 w-5" />,
  fire: <Flame className="h-5 w-5" />,
  flood: <Droplets className="h-5 w-5" />,
  flooding: <Droplets className="h-5 w-5" />,
  storm: <Wind className="h-5 w-5" />,
  hurricane: <Wind className="h-5 w-5" />,
  cyclone: <Wind className="h-5 w-5" />,
  tornado: <Wind className="h-5 w-5" />,
  earthquake: <Mountain className="h-5 w-5" />,
  landslide: <Mountain className="h-5 w-5" />,
  volcano: <Flame className="h-5 w-5" />,
  drought: <AlertCircle className="h-5 w-5" />,
  "severe storm": <CloudLightning className="h-5 w-5" />,
};

function getTypeIcon(type: string): React.ReactNode {
  const key = type.toLowerCase();
  for (const [k, icon] of Object.entries(typeIcon)) {
    if (key.includes(k)) return icon;
  }
  return <AlertTriangle className="h-5 w-5" />;
}

function getStatusBadge(status: string) {
  if (status === "ANALYZED")
    return { label: "Analyzed", cls: "bg-green-100 text-green-700 border-green-200" };
  if (status === "PROCESSING")
    return { label: "Processing", cls: "bg-blue-100 text-blue-700 border-blue-200" };
  return { label: "Ready", cls: "bg-gray-100 text-gray-600 border-gray-200" };
}

function formatDate(d: string | null) {
  if (!d) return "Date unavailable";
  try {
    return new Date(d).toLocaleDateString("en-US", {
      month: "short", day: "numeric", year: "numeric",
    });
  } catch {
    return d;
  }
}

// ── Severity config for analysis display ─────────────────────────────

const severityStyle: Record<string, { color: string; bg: string; icon: React.ReactNode }> = {
  "Severe Damage": {
    color: "text-red-600",
    bg: "bg-red-50 border-red-200",
    icon: <AlertTriangle className="h-6 w-6 text-red-500" />,
  },
  "Mild Damage": {
    color: "text-yellow-600",
    bg: "bg-yellow-50 border-yellow-200",
    icon: <AlertTriangle className="h-6 w-6 text-yellow-500" />,
  },
  "Little or No Damage": {
    color: "text-green-600",
    bg: "bg-green-50 border-green-200",
    icon: <CheckCircle className="h-6 w-6 text-green-500" />,
  },
};

function getSeverityStyle(severity: string) {
  return severityStyle[severity] || severityStyle["Mild Damage"];
}

// ── Component ────────────────────────────────────────────────────────

export default function Disasters() {
  const [events, setEvents] = useState<DisasterEventSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [analyzingId, setAnalyzingId] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<DisasterAnalysisResponse | null>(null);

  const loadEvents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDisasterEvents(50);
      setEvents(data);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Failed to fetch events";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
  }, [loadEvents]);

  const handleAnalyze = async (eventId: string) => {
    setAnalyzingId(eventId);
    setAnalysisResult(null);
    try {
      const result = await analyzeDisasterEvent(eventId);
      setAnalysisResult(result);
      // Refresh to update status badges
      loadEvents();
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Analysis failed";
      setError(msg);
    } finally {
      setAnalyzingId(null);
    }
  };

  // Stats
  const total = events.length;
  const analyzed = events.filter((e) => e.analysis_status === "ANALYZED").length;
  const pending = total - analyzed;

  return (
    <Layout>
      <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 flex items-start justify-between"
        >
          <div>
            <h1 className="text-2xl font-bold text-secondary sm:text-3xl">
              Disaster Intelligence
            </h1>
            <p className="mt-1.5 text-sm text-gray-500">
              Live events from NASA EONET — Analyze with AI to get emergency assessments
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={loadEvents}
            disabled={loading}
            className="gap-1.5"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
        </motion.div>

        {/* Statistics Panel */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="mb-8 grid grid-cols-2 gap-4 sm:grid-cols-4"
        >
          {[
            { icon: <List className="h-4 w-4" />, label: "Total Events", value: total, color: "text-primary" },
            { icon: <CheckCircle className="h-4 w-4" />, label: "Analyzed", value: analyzed, color: "text-green-600" },
            { icon: <Clock className="h-4 w-4" />, label: "Pending", value: pending, color: "text-yellow-600" },
            { icon: <AlertTriangle className="h-4 w-4" />, label: "High Priority", value: 0, color: "text-red-600" },
          ].map((stat, i) => (
            <Card key={stat.label}>
              <CardContent className="flex items-center gap-3 p-4">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/5">
                  <span className={stat.color}>{stat.icon}</span>
                </div>
                <div>
                  <p className="text-2xl font-bold text-secondary">{stat.value}</p>
                  <p className="text-xs text-gray-500">{stat.label}</p>
                </div>
              </CardContent>
            </Card>
          ))}
        </motion.div>

        {/* Error */}
        <ErrorAlert
          message={error || ""}
          visible={!!error}
          onRetry={loadEvents}
          onDismiss={() => setError(null)}
        />

        {/* Loading */}
        {loading && (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-32 w-full rounded-xl" />
            ))}
          </div>
        )}

        {/* Analysis Result */}
        <AnimatePresence>
          {analysisResult && analysisResult.success && analysisResult.analysis && (
            <motion.div
              initial={{ opacity: 0, y: 20, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: -20 }}
              className="mb-8"
            >
              <EmergencyAssessmentCard result={analysisResult} />
              {analysisResult.analysis_id && (
                <div className="mt-3">
                  <a
                    href={getReportLiveUrl(analysisResult.analysis_id)}
                    download
                    className="inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-primary/30 bg-primary/5 px-4 py-2.5 text-sm font-medium text-primary transition-colors hover:bg-primary/10"
                  >
                    <FileText className="h-4 w-4" />
                    Download PDF Report
                  </a>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Event Cards */}
        {!loading && !error && events.length === 0 && (
          <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-card p-12 text-center">
            <Satellite className="h-10 w-10 text-gray-300 mb-3" />
            <p className="text-sm font-medium text-gray-500">No disaster events available</p>
            <p className="text-xs text-gray-400 mt-1">Try refreshing or check back later.</p>
          </div>
        )}

        {!loading && events.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {events.map((event, i) => {
              const isAnalyzed = event.analysis_status === "ANALYZED";
              const statusBadge = getStatusBadge(event.analysis_status);
              return (
                <motion.div
                  key={event.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.03 }}
                >
                  <Card
                    className={`overflow-hidden transition-shadow hover:shadow-md ${
                      isAnalyzed ? "border-green-200 bg-green-50/30" : ""
                    }`}
                  >
                    <CardContent className="p-0">
                      {/* Top accent */}
                      <div
                        className={`h-1.5 w-full ${
                          isAnalyzed ? "bg-green-400" : "bg-primary/30"
                        }`}
                      />

                      <div className="p-4 space-y-3">
                        {/* Header row */}
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-2.5">
                            <div
                              className={`flex h-9 w-9 items-center justify-center rounded-lg ${
                                isAnalyzed ? "bg-green-100 text-green-600" : "bg-primary/10 text-primary"
                              }`}
                            >
                              {getTypeIcon(event.type)}
                            </div>
                            <div className="min-w-0">
                              <p className="truncate text-sm font-semibold text-secondary">
                                {event.title}
                              </p>
                              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                                <MapPin className="h-3 w-3" />
                                <span>{event.country || "Location unavailable"}</span>
                              </div>
                            </div>
                          </div>
                          <Badge className={`${statusBadge.cls} text-[10px]`}>
                            {statusBadge.label}
                          </Badge>
                        </div>

                        {/* Description */}
                        <p className="line-clamp-2 text-xs text-gray-500 leading-relaxed">
                          {event.description}
                        </p>

                        {/* Footer */}
                        <div className="flex items-center justify-between border-t border-border pt-2.5">
                          <div className="flex items-center gap-2 text-[11px] text-gray-400">
                            <Globe className="h-3 w-3" />
                            {event.source}
                            <span className="text-gray-300">|</span>
                            <Clock className="h-3 w-3" />
                            {formatDate(event.date)}
                          </div>
                          <Button
                            size="sm"
                            variant={isAnalyzed ? "outline" : "primary"}
                            onClick={() => handleAnalyze(event.id)}
                            disabled={analyzingId === event.id}
                            className="h-8 gap-1 text-xs"
                          >
                            {analyzingId === event.id ? (
                              <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                            ) : null}
                            {analyzingId === event.id
                              ? "Analyzing..."
                              : isAnalyzed
                              ? "Re-analyze"
                              : "Analyze"}
                          </Button>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </Layout>
  );
}

// ── Emergency Assessment Card ────────────────────────────────────────

function EmergencyAssessmentCard({
  result,
}: {
  result: DisasterAnalysisResponse;
}) {
  if (!result.analysis) return null;
  const { analysis, event: evt, model_used, processing_time_ms } = result;
  const s = getSeverityStyle(analysis.severity);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="overflow-hidden rounded-xl border border-blue-200 bg-white shadow-lg"
    >
      {/* Header */}
      <div className="bg-gradient-to-r from-primary to-blue-600 px-5 py-3.5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Activity className="h-5 w-5 text-white" />
            <span className="text-sm font-semibold text-white">
              Emergency Assessment
            </span>
          </div>
          <Badge className="border-white/30 bg-white/15 text-[10px] text-white">
            {evt.type}
          </Badge>
        </div>
        <p className="mt-1 text-xs text-blue-100">{evt.title}</p>
      </div>

      <div className="grid gap-5 p-5 sm:grid-cols-2">
        {/* Prediction */}
        <div className="space-y-3">
          <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
            <Cpu className="h-3.5 w-3.5" />
            AI Prediction
          </h4>

          <div className={`rounded-lg border p-3.5 ${s.bg}`}>
            <div className="flex items-center gap-3">
              {s.icon}
              <div>
                <p className="text-xs font-medium text-gray-500">Severity</p>
                <p className={`text-lg font-bold ${s.color}`}>{analysis.severity}</p>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3.5 py-2.5">
            <span className="text-xs text-gray-500">Confidence</span>
            <span className="text-sm font-bold text-secondary">
              {analysis.confidence.toFixed(1)}%
            </span>
          </div>

          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3.5 py-2.5">
            <span className="text-xs text-gray-500">Model</span>
            <span className="text-sm font-semibold text-secondary">
              {model_used || "N/A"}
            </span>
          </div>

          <div className="flex items-center justify-between rounded-lg bg-gray-50 px-3.5 py-2.5">
            <span className="text-xs text-gray-500">Country</span>
            <span className="text-sm font-semibold text-secondary">
              {evt.country || "Location unavailable"}
            </span>
          </div>
        </div>

        {/* Decision */}
        <div className="space-y-3">
          <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
            <BarChart3 className="h-3.5 w-3.5" />
            Decision Engine
          </h4>

          <div className="grid grid-cols-2 gap-2">
            <DecideBox
              label="Priority"
              value={analysis.priority}
              color={
                analysis.priority === "High"
                  ? "text-red-600 bg-red-50 border-red-200"
                  : analysis.priority === "Medium"
                  ? "text-yellow-600 bg-yellow-50 border-yellow-200"
                  : "text-green-600 bg-green-50 border-green-200"
              }
            />
            <DecideBox
              label="Infrastructure"
              value={analysis.infrastructure_impact}
              color="text-blue-600 bg-blue-50 border-blue-200"
            />
          </div>

          <div
            className={`rounded-lg border p-3 ${
              analysis.emergency_level === "Immediate Response"
                ? "border-red-200 bg-red-50"
                : analysis.emergency_level === "Field Assessment Required"
                ? "border-yellow-200 bg-yellow-50"
                : "border-green-200 bg-green-50"
            }`}
          >
            <p className="text-xs font-medium text-gray-500">Emergency Level</p>
            <p className="text-sm font-bold text-secondary">
              {analysis.emergency_level}
            </p>
          </div>

          <div className="rounded-lg border border-gray-200 bg-gray-50 p-3">
            <p className="mb-1.5 text-xs font-medium text-gray-500">
              Recommended Actions
            </p>
            <ul className="space-y-1">
              {analysis.recommended_actions.map((action, i) => (
                <li
                  key={i}
                  className="flex items-start gap-1.5 text-xs text-gray-700"
                >
                  <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary/60" />
                  {action}
                </li>
              ))}
            </ul>
          </div>

          {/* Processing time */}
          <div className="text-right text-[11px] text-gray-400">
            Processed in {(processing_time_ms || 0).toFixed(0)}ms
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function DecideBox({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className={`rounded-lg border p-2.5 ${color}`}>
      <p className="text-[10px] font-medium uppercase tracking-wider opacity-70">
        {label}
      </p>
      <p className="text-sm font-bold">{value}</p>
    </div>
  );
}

