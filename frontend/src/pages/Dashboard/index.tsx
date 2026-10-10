import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend,
} from "recharts";
import {
  Activity, BarChart3, Brain, Clock, Cpu, Database, FileImage, Globe,
  LayoutDashboard, RefreshCw, Satellite, Server, Shield, AlertTriangle,
  Zap, CheckCircle, XCircle, ExternalLink,
} from "lucide-react";
import { Layout } from "@/components/Layout";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import LoadingSpinner from "@/components/LoadingSpinner";
import ErrorAlert from "@/components/ErrorAlert";
import {
  fetchDashboardSummary,
  fetchDashboardStatistics,
  fetchDashboardCharts,
  fetchDashboardTimeline,
  checkHealth,
  fetchDisasterEvents,
} from "@/services/api";
import type {
  DashboardSummary, DashboardStatistics, DashboardChartData, TimelineEntry,
  DisasterEventSummary, HealthResponse,
} from "@/types";

// ── Colors ──────────────────────────────────────────────────────────

const CHART_COLORS = ["#2563eb", "#f59e0b", "#ef4444", "#10b981", "#8b5cf6", "#ec4899", "#06b6d4"];
const SEVERITY_COLORS: Record<string, string> = {
  "Little or No Damage": "#10b981",
  "Mild Damage": "#f59e0b",
  "Severe Damage": "#ef4444",
};
const PRIORITY_COLORS: Record<string, string> = {
  Low: "#10b981", Medium: "#f59e0b", High: "#ef4444",
};

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.04 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0 },
};

// ── Main Component ──────────────────────────────────────────────────

export default function Dashboard() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const [summary, setSummary] = useState<DashboardSummary | null>(null);
  const [stats, setStats] = useState<DashboardStatistics | null>(null);
  const [charts, setCharts] = useState<DashboardChartData | null>(null);
  const [timeline, setTimeline] = useState<TimelineEntry[]>([]);
  const [events, setEvents] = useState<DisasterEventSummary[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const loadAll = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, st, c, t, e, h] = await Promise.all([
        fetchDashboardSummary(),
        fetchDashboardStatistics(),
        fetchDashboardCharts(),
        fetchDashboardTimeline(20),
        fetchDisasterEvents(10),
        checkHealth(),
      ]);
      setSummary(s);
      setStats(st);
      setCharts(c);
      setTimeline(t);
      setEvents(e);
      setHealth(h);
      setLastUpdated(new Date());
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Failed to load dashboard";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadAll();
  }, [loadAll]);

  const isLoading = loading && !summary;

  return (
    <Layout>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* ── Section 1: Hero Banner ────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 overflow-hidden rounded-xl bg-gradient-to-r from-primary via-blue-600 to-blue-700 shadow-lg"
        >
          <div className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2.5">
                <LayoutDashboard className="h-6 w-6 text-white/90" />
                <h1 className="text-xl font-bold text-white sm:text-2xl">
                  Emergency Operations Dashboard
                </h1>
              </div>
              <div className="flex items-center gap-3 text-sm text-blue-100">
                <span className="flex items-center gap-1">
                  <Activity className="h-3.5 w-3.5" />
                  {health?.status === "healthy" ? "All Systems Operational" : "Checking..."}
                </span>
                {lastUpdated && (
                  <span className="flex items-center gap-1">
                    <Clock className="h-3.5 w-3.5" />
                    Updated {lastUpdated.toLocaleTimeString()}
                  </span>
                )}
              </div>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={loadAll}
              disabled={loading}
              className="gap-1.5 bg-white/15 text-white hover:bg-white/25"
            >
              <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
              Refresh
            </Button>
          </div>
        </motion.div>

        {/* Error */}
        <ErrorAlert
          message={error || ""}
          visible={!!error}
          onRetry={loadAll}
          onDismiss={() => setError(null)}
        />

        {isLoading ? (
          <LoadingGrid />
        ) : (
          <motion.div variants={containerVariants} initial="hidden" animate="visible" className="space-y-8">
            {/* ── Section 2: Statistics Cards ───────────────────────── */}
            <StatsSection summary={summary} stats={stats} />

            {/* ── Section 3: Charts ────────────────────────────────── */}
            <ChartsSection charts={charts} />

            {/* ── Section 4: Activity Timeline ─────────────────────── */}
            <TimelineSection timeline={timeline} />

            {/* ── Section 5: Recent Live Events ────────────────────── */}
            <EventsSection events={events} />

            {/* ── Section 6: Quick Actions ─────────────────────────── */}
            <QuickActions />

            {/* ── Section 7: System Health ─────────────────────────── */}
            <HealthSection health={health} />
          </motion.div>
        )}
      </div>
    </Layout>
  );
}

// ── Loading Grid ────────────────────────────────────────────────────

function LoadingGrid() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
        {Array.from({ length: 7 }).map((_, i) => (
          <Skeleton key={i} className="h-24 rounded-xl" />
        ))}
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
      <Skeleton className="h-80 rounded-xl" />
    </div>
  );
}

// ── Section 2: Statistics ───────────────────────────────────────────

function StatsSection({ summary, stats }: { summary: DashboardSummary | null; stats: DashboardStatistics | null }) {
  const items = [
    { icon: <Activity className="h-4 w-4" />, label: "Total Analyses", value: summary?.total_analyses ?? 0, color: "text-primary" },
    { icon: <Brain className="h-4 w-4" />, label: "Manual", value: summary?.manual ?? 0, color: "text-purple-600" },
    { icon: <Satellite className="h-4 w-4" />, label: "Live Events", value: summary?.live ?? 0, color: "text-blue-600" },
    { icon: <FileImage className="h-4 w-4" />, label: "Batch Files", value: summary?.batch ?? 0, color: "text-emerald-600" },
    { icon: <AlertTriangle className="h-4 w-4" />, label: "High Priority", value: summary?.high_priority ?? 0, color: "text-red-600" },
    { icon: <Cpu className="h-4 w-4" />, label: "Avg Confidence", value: summary ? `${summary.average_confidence.toFixed(1)}%` : "0%", color: "text-cyan-600" },
    { icon: <Clock className="h-4 w-4" />, label: "Avg Time", value: summary ? `${summary.average_processing_time_ms.toFixed(0)}ms` : "0ms", color: "text-amber-600" },
  ];

  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <Activity className="mr-1.5 inline h-4 w-4" />
        Analysis Overview
      </h2>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
        {items.map((stat) => (
          <motion.div key={stat.label} whileHover={{ y: -2 }} transition={{ type: "spring", stiffness: 300 }}>
            <Card className="h-full">
              <CardContent className="flex flex-col items-center justify-center gap-1.5 p-4 text-center">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/5">
                  <span className={stat.color}>{stat.icon}</span>
                </div>
                <p className="text-xl font-bold text-secondary">{stat.value}</p>
                <p className="text-[11px] text-gray-500">{stat.label}</p>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
      {stats && (
        <p className="mt-2 text-center text-xs text-gray-400">
          Most common: <strong>{stats.most_common_disaster_type}</strong> type &middot;{" "}
          <strong>{stats.most_common_severity}</strong> severity
        </p>
      )}
    </motion.div>
  );
}

// ── Section 3: Charts ───────────────────────────────────────────────

function ChartsSection({ charts }: { charts: DashboardChartData | null }) {
  if (!charts) return null;

  const chartConfigs = [
    {
      title: "Severity Distribution", data: charts.severity,
      render: (d: { name: string; value: number }[]) => (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={d} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" height={40} />
            <YAxis tick={{ fontSize: 10 }} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {d.map((entry) => (
                <Cell key={entry.name} fill={SEVERITY_COLORS[entry.name] || CHART_COLORS[0]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ),
    },
    {
      title: "Priority Distribution", data: charts.priority,
      render: (d: { name: string; value: number }[]) => (
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={d} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, percent }: any) => `${String(name ?? "")} ${((percent ?? 0) * 100).toFixed(0)}%`} labelLine={false}>
              {d.map((entry) => (
                <Cell key={entry.name} fill={PRIORITY_COLORS[entry.name] || CHART_COLORS[0]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      ),
    },
    {
      title: "Disaster Types", data: charts.disaster_types,
      render: (d: { name: string; value: number }[]) => (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={d} layout="vertical" margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
            <XAxis type="number" tick={{ fontSize: 10 }} />
            <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={90} />
            <Tooltip contentStyle={{ fontSize: 12 }} />
            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
              {d.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      ),
    },
    {
      title: "Analysis Modes", data: charts.analysis_modes,
      render: (d: { name: string; value: number }[]) => (
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={d} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label={({ name, percent }: any) => `${String(name ?? "")} ${((percent ?? 0) * 100).toFixed(0)}%`} labelLine={false}>
              {d.map((_, i) => (
                <Cell key={i} fill={CHART_COLORS[i % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      ),
    },
  ];

  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <BarChart3 className="mr-1.5 inline h-4 w-4" />
        Analytics & Charts
      </h2>
      <div className="grid gap-4 sm:grid-cols-2">
        {chartConfigs.map((cfg) => (
          <motion.div key={cfg.title} initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.4 }}>
            <Card>
              <CardContent className="p-4">
                <h3 className="mb-3 text-xs font-semibold text-gray-500">{cfg.title}</h3>
                {cfg.data.length > 0 ? cfg.render(cfg.data) : (
                  <div className="flex h-[220px] items-center justify-center text-xs text-gray-400">No data</div>
                )}
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

// ── Section 4: Timeline ─────────────────────────────────────────────

function TimelineSection({ timeline }: { timeline: TimelineEntry[] }) {
  const severityBadge: Record<string, string> = {
    "Severe Damage": "bg-red-100 text-red-700 border-red-200",
    "Mild Damage": "bg-yellow-100 text-yellow-700 border-yellow-200",
    "Little or No Damage": "bg-green-100 text-green-700 border-green-200",
  };
  const priorityBadge: Record<string, string> = {
    High: "bg-red-50 text-red-600 border-red-100",
    Medium: "bg-yellow-50 text-yellow-600 border-yellow-100",
    Low: "bg-green-50 text-green-600 border-green-100",
  };
  const typeIcon: Record<string, React.ReactNode> = {
    manual: <Brain className="h-3.5 w-3.5" />,
    live: <Satellite className="h-3.5 w-3.5" />,
    batch: <FileImage className="h-3.5 w-3.5" />,
  };
  const typeLabel: Record<string, string> = { manual: "Manual", live: "Live Event", batch: "Batch" };

  function formatTime(ts: string) {
    try {
      const d = new Date(ts);
      return d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    } catch { return ts; }
  }

  function truncateId(id: string) {
    if (!id || id.length <= 8) return id;
    return id.slice(0, 8) + "\u2026";
  }

  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <Activity className="mr-1.5 inline h-4 w-4" />
        Recent Activity
      </h2>
      <Card>
        <CardContent className="divide-y divide-border p-0">
          {timeline.length === 0 ? (
            <div className="flex flex-col items-center justify-center p-12 text-center">
              <Activity className="mb-2 h-8 w-8 text-gray-200" />
              <p className="text-sm text-gray-500">No recent activity</p>
              <p className="text-xs text-gray-400 mt-1">Analyses will appear here as they are performed</p>
            </div>
          ) : (
            timeline.map((entry, i) => (
              <motion.div
                key={`${entry.id}-${i}`}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.02 }}
                className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-gray-50/50"
              >
                <div className={`flex h-8 w-8 items-center justify-center rounded-full ${
                  entry.type === "live" ? "bg-blue-100 text-blue-600"
                    : entry.type === "batch" ? "bg-emerald-100 text-emerald-600"
                    : "bg-purple-100 text-purple-600"
                }`}>
                  {typeIcon[entry.type] || <Zap className="h-3.5 w-3.5" />}
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-medium text-secondary">
                      {entry.event_title || entry.filename || truncateId(entry.id)}
                    </span>
                    <Badge className="bg-gray-100 text-gray-500 border-gray-200 text-[10px]">
                      {typeLabel[entry.type] || entry.type}
                    </Badge>
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] text-gray-400">
                    <span>{formatTime(entry.timestamp)}</span>
                    {entry.severity && (
                      <Badge className={`${severityBadge[entry.severity] || ""} text-[10px]`}>
                        {entry.severity}
                      </Badge>
                    )}
                    {entry.priority && (
                      <Badge className={`${priorityBadge[entry.priority] || ""} text-[10px]`}>
                        {entry.priority}
                      </Badge>
                    )}
                    {entry.analysis_mode && <span>{entry.analysis_mode}</span>}
                    {entry.processing_time_ms > 0 && <span>{entry.processing_time_ms.toFixed(0)}ms</span>}
                  </div>
                </div>
                {!entry.success && (
                  <XCircle className="h-4 w-4 flex-shrink-0 text-red-400" />
                )}
              </motion.div>
            ))
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ── Section 5: Events ───────────────────────────────────────────────

function EventsSection({ events }: { events: DisasterEventSummary[] }) {
  const statusBadge: Record<string, string> = {
    ANALYZED: "bg-green-100 text-green-700 border-green-200",
    PROCESSING: "bg-blue-100 text-blue-700 border-blue-200",
    READY_FOR_AI: "bg-gray-100 text-gray-600 border-gray-200",
  };

  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <Satellite className="mr-1.5 inline h-4 w-4" />
        Recent Live Events
      </h2>
      <Card>
        <CardContent className="divide-y divide-border p-0">
          {events.length === 0 ? (
            <div className="flex flex-col items-center justify-center p-12 text-center">
              <Satellite className="mb-2 h-8 w-8 text-gray-200" />
              <p className="text-sm text-gray-500">No live events</p>
            </div>
          ) : (
            events.slice(0, 6).map((evt) => (
              <div key={evt.id} className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-gray-50/50">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-100 text-blue-600">
                  <Globe className="h-4 w-4" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-secondary">{evt.title}</p>
                  <p className="text-xs text-gray-400">{evt.country || "Location unavailable"} &middot; {evt.type}</p>
                </div>
                <Badge className={`${statusBadge[evt.analysis_status] || statusBadge["READY_FOR_AI"]} text-[10px]`}>
                  {evt.analysis_status === "ANALYZED" ? "Analyzed" : evt.analysis_status === "PROCESSING" ? "Processing" : "Ready"}
                </Badge>
              </div>
            ))
          )}
        </CardContent>
      </Card>
      <div className="mt-2 text-right">
        <Link to="/disasters" className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:text-primary/80">
          View all events <ExternalLink className="h-3 w-3" />
        </Link>
      </div>
    </motion.div>
  );
}

// ── Section 6: Quick Actions ────────────────────────────────────────

const actions = [
  { icon: <Brain className="h-5 w-5" />, label: "Manual Analysis", desc: "Upload an image or enter text for AI damage assessment", to: "/analysis", color: "bg-purple-500" },
  { icon: <Satellite className="h-5 w-5" />, label: "Disaster Intelligence", desc: "Browse live events and run AI analyses", to: "/disasters", color: "bg-blue-500" },
  { icon: <FileImage className="h-5 w-5" />, label: "Batch Analysis", desc: "Upload multiple files for bulk processing", to: "/batch", color: "bg-emerald-500" },
];

function QuickActions() {
  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <Zap className="mr-1.5 inline h-4 w-4" />
        Quick Actions
      </h2>
      <div className="grid gap-4 sm:grid-cols-3">
        {actions.map((a) => (
          <Link key={a.to} to={a.to}>
            <motion.div whileHover={{ y: -3, boxShadow: "0 4px 12px rgba(0,0,0,0.08)" }} transition={{ type: "spring", stiffness: 300 }}>
              <Card className="h-full cursor-pointer transition-colors hover:border-primary/30">
                <CardContent className="flex items-start gap-4 p-5">
                  <div className={`flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-xl text-white ${a.color}`}>
                    {a.icon}
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-secondary">{a.label}</p>
                    <p className="mt-0.5 text-xs text-gray-500">{a.desc}</p>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </Link>
        ))}
      </div>
    </motion.div>
  );
}

// ── Section 7: System Health ────────────────────────────────────────

function HealthSection({ health }: { health: HealthResponse | null }) {
  const items = [
    { icon: <Server className="h-4 w-4" />, label: "Backend Status", ok: health?.status === "healthy", detail: health?.status || "unknown" },
    { icon: <Brain className="h-4 w-4" />, label: "AI Models", ok: !!health?.models_loaded, detail: health?.models_loaded ? `${health.models_loaded} models loaded` : "Checking..." },
    { icon: <Satellite className="h-4 w-4" />, label: "Provider Status", ok: true, detail: "NASA EONET" },
    { icon: <Database className="h-4 w-4" />, label: "Cache Status", ok: true, detail: "In-memory store active" },
    { icon: <Globe className="h-4 w-4" />, label: "API Status", ok: true, detail: "v1.0.0" },
  ];

  return (
    <motion.div variants={itemVariants}>
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-400">
        <Shield className="mr-1.5 inline h-4 w-4" />
        System Health
      </h2>
      <Card>
        <CardContent className="p-4">
          <div className="grid gap-3 sm:grid-cols-5">
            {items.map((item) => (
              <div key={item.label} className="flex items-center gap-3 rounded-lg border border-border bg-gray-50/50 p-3">
                <div className={`flex h-9 w-9 items-center justify-center rounded-full ${item.ok ? "bg-green-100 text-green-600" : "bg-red-100 text-red-600"}`}>
                  {item.icon}
                </div>
                <div>
                  <p className="text-xs font-medium text-secondary">{item.label}</p>
                  <p className="flex items-center gap-1 text-[11px] text-gray-400">
                    <span className={`inline-block h-1.5 w-1.5 rounded-full ${item.ok ? "bg-green-500" : "bg-red-500"}`} />
                    {item.detail}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
