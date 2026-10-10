import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  Search, Filter, X, Download, Eye, ChevronLeft, ChevronRight,
  BarChart3, Activity, Brain, Clock, SortAsc,
} from "lucide-react";
import { Layout } from "@/components/Layout";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import ErrorAlert from "@/components/ErrorAlert";
import {
  fetchAnalysisCenter,
  fetchAnalysisCenterStatistics,
  fetchAnalysisCenterItem,
  getAnalysisCenterExportUrl,
  getReportManualUrl,
  getReportLiveUrl,
} from "@/services/api";
import type { AnalysisCenterRecord, AnalysisCenterResponse, AnalysisCenterStatistics } from "@/types";

const SEVERITY_COLORS: Record<string, string> = {
  "Severe Damage": "bg-red-100 text-red-700 border-red-200",
  "Mild Damage": "bg-yellow-100 text-yellow-700 border-yellow-200",
  "Little or No Damage": "bg-green-100 text-green-700 border-green-200",
};

const PRIORITY_BADGE: Record<string, string> = {
  High: "bg-red-50 text-red-600 border-red-100",
  Medium: "bg-yellow-50 text-yellow-600 border-yellow-100",
  Low: "bg-green-50 text-green-600 border-green-100",
  Critical: "bg-purple-50 text-purple-600 border-purple-100",
};

const TYPE_BADGE: Record<string, string> = {
  manual: "bg-purple-100 text-purple-700 border-purple-200",
  live: "bg-blue-100 text-blue-700 border-blue-200",
  batch: "bg-emerald-100 text-emerald-700 border-emerald-200",
};

const containerVariants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.04 } },
};
const itemVariants = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0 },
};

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

export default function AnalysisCenter() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [data, setData] = useState<AnalysisCenterResponse | null>(null);
  const [stats, setStats] = useState<AnalysisCenterStatistics | null>(null);
  const [detail, setDetail] = useState<AnalysisCenterRecord | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState("newest");
  const [query, setQuery] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [filterType, setFilterType] = useState("");
  const [filterSeverity, setFilterSeverity] = useState("");
  const [filterPriority, setFilterPriority] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  const loadData = useCallback(async (p: number) => {
    setLoading(true);
    setError(null);
    try {
      const params: Record<string, any> = { page: p, per_page: 20, sort_by: sortBy };
      if (query) params.query = query;
      if (filterType) params.record_type = filterType;
      if (filterSeverity) params.severity = filterSeverity;
      if (filterPriority) params.priority = filterPriority;
      const result = await fetchAnalysisCenter(params);
      setData(result);
    } catch (e: any) {
      setError(e?.response?.data?.detail || e?.message || "Failed to load analyses");
    } finally {
      setLoading(false);
    }
  }, [query, sortBy, filterType, filterSeverity, filterPriority]);

  const loadStats = useCallback(async () => {
    try {
      const s = await fetchAnalysisCenterStatistics();
      setStats(s);
    } catch { /* non-critical */ }
  }, []);

  useEffect(() => { loadData(page); }, [page, loadData]);
  useEffect(() => { loadStats(); }, [loadStats]);

  const handleSearch = () => {
    setQuery(searchInput);
    setPage(1);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSearch();
  };

  const clearFilters = () => {
    setFilterType("");
    setFilterSeverity("");
    setFilterPriority("");
    setQuery("");
    setSearchInput("");
    setPage(1);
  };

  const openDetail = async (id: string) => {
    setDetailLoading(true);
    setDetail(null);
    try {
      const record = await fetchAnalysisCenterItem(id);
      setDetail(record);
    } catch (e: any) {
      setError(e?.message || "Failed to load detail");
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <Layout>
      <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 overflow-hidden rounded-xl bg-gradient-to-r from-primary via-blue-600 to-blue-700 shadow-lg"
        >
          <div className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
            <div className="space-y-1">
              <div className="flex items-center gap-2.5">
                <BarChart3 className="h-6 w-6 text-white/90" />
                <h1 className="text-xl font-bold text-white sm:text-2xl">Analysis Center</h1>
              </div>
              <p className="text-sm text-blue-100">Browse, search, and export all completed analyses</p>
            </div>
            <div className="flex items-center gap-2">
              {data && data.total_items > 0 && (
                <Button
                  variant="secondary"
                  size="sm"
                  className="gap-1.5 bg-white/15 text-white hover:bg-white/25"
                  onClick={() => window.open(getAnalysisCenterExportUrl(), "_blank")}
                >
                  <Download className="h-4 w-4" />
                  Export CSV
                </Button>
              )}
              <Button
                variant="secondary"
                size="sm"
                className="gap-1.5 bg-white/15 text-white hover:bg-white/25"
                onClick={() => { loadData(page); loadStats(); }}
                disabled={loading}
              >
                <Activity className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
                Refresh
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Error */}
        <ErrorAlert message={error || ""} visible={!!error} onDismiss={() => setError(null)} onRetry={() => loadData(page)} />

        {/* Search + Filter bar */}
        <motion.div variants={itemVariants} className="mb-6 space-y-3">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search by ID, disaster type, country, description..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                onKeyDown={handleKeyDown}
                className="h-10 w-full rounded-lg border border-border bg-white pl-9 pr-4 text-sm outline-none focus:border-primary/50 focus:ring-2 focus:ring-primary/10"
              />
            </div>
            <Button variant="outline" size="sm" onClick={handleSearch} className="h-10 gap-1.5">
              <Search className="h-4 w-4" />
              Search
            </Button>
            <Button
              variant={showFilters ? "primary" : "outline"}
              size="sm"
              onClick={() => setShowFilters(!showFilters)}
              className="h-10 gap-1.5"
            >
              <Filter className="h-4 w-4" />
              Filters
            </Button>
            <div className="flex items-center gap-1">
              <SortAsc className="h-4 w-4 text-gray-400" />
              <select
                value={sortBy}
                onChange={(e) => { setSortBy(e.target.value); setPage(1); }}
                className="h-10 rounded-lg border border-border bg-white px-2 text-sm outline-none focus:border-primary/50"
              >
                <option value="newest">Newest</option>
                <option value="oldest">Oldest</option>
                <option value="highest_confidence">Highest Confidence</option>
                <option value="lowest_confidence">Lowest Confidence</option>
                <option value="highest_priority">Highest Priority</option>
              </select>
            </div>
          </div>

          {showFilters && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="flex flex-wrap items-center gap-3 rounded-lg border border-border bg-gray-50/50 p-3"
            >
              <div className="flex items-center gap-2">
                <label className="text-xs font-medium text-gray-500">Type:</label>
                <select
                  value={filterType}
                  onChange={(e) => { setFilterType(e.target.value); setPage(1); }}
                  className="h-8 rounded-md border border-border bg-white px-2 text-xs outline-none"
                >
                  <option value="">All</option>
                  <option value="manual">Manual</option>
                  <option value="live">Live</option>
                  <option value="batch">Batch</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs font-medium text-gray-500">Severity:</label>
                <select
                  value={filterSeverity}
                  onChange={(e) => { setFilterSeverity(e.target.value); setPage(1); }}
                  className="h-8 rounded-md border border-border bg-white px-2 text-xs outline-none"
                >
                  <option value="">All</option>
                  <option value="Severe Damage">Severe</option>
                  <option value="Mild Damage">Mild</option>
                  <option value="Little or No Damage">Little/None</option>
                </select>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-xs font-medium text-gray-500">Priority:</label>
                <select
                  value={filterPriority}
                  onChange={(e) => { setFilterPriority(e.target.value); setPage(1); }}
                  className="h-8 rounded-md border border-border bg-white px-2 text-xs outline-none"
                >
                  <option value="">All</option>
                  <option value="Critical">Critical</option>
                  <option value="High">High</option>
                  <option value="Medium">Medium</option>
                  <option value="Low">Low</option>
                </select>
              </div>
              {(filterType || filterSeverity || filterPriority || query) && (
                <Button variant="ghost" size="sm" onClick={clearFilters} className="h-8 gap-1 text-xs">
                  <X className="h-3 w-3" />
                  Clear
                </Button>
              )}
            </motion.div>
          )}
        </motion.div>

        {/* Statistics summary */}
        {stats && (
          <motion.div variants={itemVariants} className="mb-6">
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-secondary">{stats.total_analyses}</p>
                  <p className="text-[10px] text-gray-500">Total</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-purple-600">{stats.by_type.manual || 0}</p>
                  <p className="text-[10px] text-gray-500">Manual</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-blue-600">{stats.by_type.live || 0}</p>
                  <p className="text-[10px] text-gray-500">Live</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-emerald-600">{stats.by_type.batch || 0}</p>
                  <p className="text-[10px] text-gray-500">Batch</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-cyan-600">{stats.average_confidence.toFixed(1)}%</p>
                  <p className="text-[10px] text-gray-500">Avg Conf</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 text-center">
                  <p className="text-lg font-bold text-amber-600">{stats.average_processing_time_ms.toFixed(0)}ms</p>
                  <p className="text-[10px] text-gray-500">Avg Time</p>
                </CardContent>
              </Card>
            </div>
          </motion.div>
        )}

        {/* Table */}
        {loading && !data ? (
          <div className="space-y-3">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-16 rounded-lg" />
            ))}
          </div>
        ) : data && data.items.length > 0 ? (
          <motion.div variants={containerVariants} initial="hidden" animate="visible">
            <Card>
              <CardContent className="divide-y divide-border p-0">
                {/* Header row */}
                <div className="hidden items-center gap-3 px-4 py-2.5 text-[11px] font-semibold uppercase tracking-wider text-gray-400 sm:flex">
                  <div className="w-24">Type</div>
                  <div className="flex-1 min-w-0">Event / File</div>
                  <div className="w-24 text-center">Severity</div>
                  <div className="w-20 text-center">Confidence</div>
                  <div className="w-20 text-center">Priority</div>
                  <div className="w-14 text-center">Time</div>
                  <div className="w-20 text-right">Actions</div>
                </div>
                {data.items.map((record, i) => (
                  <motion.div
                    key={record.analysis_id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.02 }}
                    className="flex flex-col gap-1 px-4 py-3 transition-colors hover:bg-gray-50/50 sm:flex-row sm:items-center sm:gap-3"
                  >
                    <div className="flex items-center gap-2 sm:w-24">
                      <Badge className={`${TYPE_BADGE[record.record_type] || "bg-gray-100 text-gray-600"} text-[10px]`}>
                        {record.record_type}
                      </Badge>
                      <Badge className={`${record.analysis_mode === "text-only" ? "bg-gray-100 text-gray-500" : "bg-indigo-50 text-indigo-600"} border text-[10px]`}>
                        {record.analysis_mode || "—"}
                      </Badge>
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-secondary">
                        {record.event_title || record.filename || truncateId(record.analysis_id)}
                      </p>
                      <p className="truncate text-[11px] text-gray-400">
                        {record.disaster_type && <span className="font-medium">{record.disaster_type}</span>}
                        {record.country && <span> &middot; {record.country}</span>}
                        {record.description && <span className="ml-1">&middot; {record.description.slice(0, 60)}</span>}
                        <span className="ml-1 text-gray-300">{formatTime(record.timestamp)}</span>
                      </p>
                    </div>
                    <div className="flex items-center gap-2 sm:w-24 sm:justify-center">
                      {record.severity ? (
                        <Badge className={`${SEVERITY_COLORS[record.severity] || ""} text-[10px]`}>
                          {record.severity === "Little or No Damage" ? "Little/None" : record.severity}
                        </Badge>
                      ) : (
                        <span className="text-[10px] text-gray-300">—</span>
                      )}
                    </div>
                    <div className="text-center text-sm font-medium text-secondary sm:w-20">
                      {record.confidence != null ? `${record.confidence.toFixed(1)}%` : "\u2014"}
                    </div>
                    <div className="text-center sm:w-20">
                      {record.priority ? (
                        <Badge className={`${PRIORITY_BADGE[record.priority] || ""} text-[10px]`}>
                          {record.priority}
                        </Badge>
                      ) : (
                        <span className="text-[10px] text-gray-300">—</span>
                      )}
                    </div>
                    <div className="text-center text-[11px] text-gray-400 sm:w-14">
                      {record.processing_time_ms > 0 ? `${record.processing_time_ms.toFixed(0)}ms` : "—"}
                    </div>
                    <div className="flex items-center justify-end gap-1 sm:w-20">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 w-7 p-0"
                        onClick={() => openDetail(record.analysis_id)}
                        title="View details"
                      >
                        <Eye className="h-3.5 w-3.5" />
                      </Button>
                    </div>
                  </motion.div>
                ))}
              </CardContent>
            </Card>

            {/* Pagination */}
            <div className="mt-4 flex items-center justify-between">
              <p className="text-xs text-gray-400">
                Showing {(data.page - 1) * data.per_page + 1}–{Math.min(data.page * data.per_page, data.total_items)} of {data.total_items}
              </p>
              <div className="flex items-center gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!data.has_prev}
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  className="h-8 w-8 p-0"
                >
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                {Array.from({ length: Math.min(data.total_pages, 5) }).map((_, i) => {
                  const startPage = Math.max(1, data.page - 2);
                  const pg = startPage + i;
                  if (pg > data.total_pages) return null;
                  return (
                    <Button
                      key={pg}
                      variant={pg === data.page ? "primary" : "outline"}
                      size="sm"
                      onClick={() => setPage(pg)}
                      className="h-8 w-8 p-0 text-xs"
                    >
                      {pg}
                    </Button>
                  );
                })}
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!data.has_next}
                  onClick={() => setPage((p) => p + 1)}
                  className="h-8 w-8 p-0"
                >
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div variants={itemVariants} className="flex flex-col items-center justify-center py-16 text-center">
            <BarChart3 className="mb-3 h-12 w-12 text-gray-200" />
            <p className="text-sm font-medium text-gray-500">No analyses found</p>
            <p className="mt-1 text-xs text-gray-400">
              {query || filterType || filterSeverity || filterPriority
                ? "Try adjusting your search or filters"
                : "Run an analysis to see results here"}
            </p>
          </motion.div>
        )}
      </div>

      {/* Detail Modal */}
      {detail && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={() => setDetail(null)}>
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="relative max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-xl bg-white shadow-xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 flex items-center justify-between border-b border-border bg-white px-6 py-4">
              <h2 className="text-base font-semibold text-secondary">Analysis Detail</h2>
              <button onClick={() => setDetail(null)} className="rounded p-1 text-gray-400 hover:text-gray-600">
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="space-y-4 p-6">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">ID</p>
                  <p className="font-mono text-xs text-secondary">{detail.analysis_id}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Timestamp</p>
                  <p className="text-secondary">{formatTime(detail.timestamp)}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Type</p>
                  <Badge className={`${TYPE_BADGE[detail.record_type] || ""} mt-0.5`}>{detail.record_type}</Badge>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Source</p>
                  <p className="text-secondary">{detail.source || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Analysis Mode</p>
                  <Badge className="mt-0.5 border bg-gray-50 text-gray-600">{detail.analysis_mode || "—"}</Badge>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Model</p>
                  <p className="font-mono text-xs text-secondary">{detail.model_used || "—"}</p>
                </div>
              </div>

              <hr className="border-border" />

              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Severity</p>
                  {detail.severity ? (
                    <Badge className={`${SEVERITY_COLORS[detail.severity] || ""} mt-1`}>{detail.severity}</Badge>
                  ) : <p className="mt-1 text-xs text-gray-300">—</p>}
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Confidence</p>
                  <p className="mt-1 text-lg font-bold text-secondary">
                    {detail.confidence != null ? `${detail.confidence.toFixed(1)}%` : "\u2014"}
                  </p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Priority</p>
                  {detail.priority ? (
                    <Badge className={`${PRIORITY_BADGE[detail.priority] || ""} mt-1`}>{detail.priority}</Badge>
                  ) : <p className="mt-1 text-xs text-gray-300">—</p>}
                </div>
              </div>

              <hr className="border-border" />

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Disaster Type</p>
                  <p className="text-secondary">{detail.disaster_type || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Country</p>
                  <p className="text-secondary">{detail.country || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Infrastructure Impact</p>
                  <p className="text-secondary">{detail.infrastructure_impact || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Emergency Level</p>
                  <p className="text-secondary">{detail.emergency_level || "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Processing Time</p>
                  <p className="text-secondary">{detail.processing_time_ms > 0 ? `${detail.processing_time_ms.toFixed(0)}ms` : "—"}</p>
                </div>
                <div>
                  <p className="text-[11px] font-medium uppercase tracking-wider text-gray-400">Status</p>
                  <Badge className={detail.success ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}>
                    {detail.success ? "Success" : "Failed"}
                  </Badge>
                </div>
              </div>

              {detail.error && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3">
                  <p className="text-[11px] font-medium text-red-700">Error</p>
                  <p className="text-xs text-red-600">{detail.error}</p>
                </div>
              )}

              {detail.recommended_actions && detail.recommended_actions.length > 0 && (
                <div>
                  <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-gray-400">Recommended Actions</p>
                  <ul className="space-y-1">
                    {detail.recommended_actions.map((action, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                        <span className="mt-1 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                        {action}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {detail.has_explanation && (
                <div className="rounded-lg border border-blue-200 bg-blue-50 p-3 text-sm text-blue-700">
                  This analysis has an AI Explanation overlay image available.
                </div>
              )}

              {/* Actions */}
              <div className="flex flex-wrap gap-2 pt-2">
                {(detail.record_type === "manual" || detail.record_type === "batch") && (
                  <Button variant="outline" size="sm" className="gap-1.5" onClick={() => window.open(getReportManualUrl(detail.analysis_id), "_blank")}>
                    <Download className="h-3.5 w-3.5" />
                    Report PDF
                  </Button>
                )}
                {detail.record_type === "live" && (
                  <Button variant="outline" size="sm" className="gap-1.5" onClick={() => window.open(getReportLiveUrl(detail.analysis_id), "_blank")}>
                    <Download className="h-3.5 w-3.5" />
                    Report PDF
                  </Button>
                )}
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Detail Loading */}
      {detailLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <Card className="w-full max-w-md">
            <CardContent className="flex items-center justify-center p-8">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary/20 border-t-primary" />
                <p className="text-sm text-gray-500">Loading details...</p>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </Layout>
  );
}
