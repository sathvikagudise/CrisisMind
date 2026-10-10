import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileImage,
  Table,
  FileText,
  Download,
  Play,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Clock,
  Cpu,
  BarChart3,
  List,
  RefreshCw,
  Layers,
  ChevronDown,
  ChevronUp,
  Trash2,
} from "lucide-react";
import { Layout } from "@/components/Layout";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import ErrorAlert from "@/components/ErrorAlert";
import {
  uploadBatch,
  startBatch,
  getBatchStatus,
  getBatchResults,
  getReportBatchUrl,
  getReportCsvUrl,
} from "@/services/api";
import type {
  BatchUploadResponse,
  BatchStatusResponse,
  BatchResultsResponse,
  BatchFileResult,
  BatchSummary,
} from "@/types";

type Phase = "upload" | "processing" | "results";

const severityColors: Record<string, string> = {
  "Little or No Damage": "bg-green-500",
  "Mild Damage": "bg-yellow-500",
  "Severe Damage": "bg-red-500",
};

const priorityColors: Record<string, string> = {
  Low: "bg-green-100 text-green-700 border-green-200",
  Medium: "bg-yellow-100 text-yellow-700 border-yellow-200",
  High: "bg-red-100 text-red-700 border-red-200",
};

function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`;
}

export default function Batch() {
  const [phase, setPhase] = useState<Phase>("upload");
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batchId, setBatchId] = useState<string | null>(null);
  const [status, setStatus] = useState<BatchStatusResponse | null>(null);
  const [results, setResults] = useState<BatchResultsResponse | null>(null);
  const [sortKey, setSortKey] = useState<string>("filename");
  const [sortAsc, setSortAsc] = useState(true);
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const imageInputRef = useRef<HTMLInputElement>(null);
  const csvInputRef = useRef<HTMLInputElement>(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async (id: string) => {
    try {
      const s = await getBatchStatus(id);
      setStatus(s);
      if (s.status === "completed" || s.status === "failed") {
        stopPolling();
        const r = await getBatchResults(id);
        setResults(r);
        setPhase("results");
      }
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Failed to poll status";
      setError(msg);
    }
  }, [stopPolling]);

  const handleUpload = async () => {
    if (imageFiles.length === 0) {
      setError("Select at least one image file to upload");
      return;
    }
    setUploading(true);
    setError(null);
    try {
      const resp: BatchUploadResponse = await uploadBatch(imageFiles, csvFile);
      setBatchId(resp.batch_id);
      await startBatch(resp.batch_id);
      setPhase("processing");
      setStatus({
        batch_id: resp.batch_id,
        status: "processing",
        total_files: resp.total_files,
        completed: 0,
        failed: 0,
        pending: resp.total_files,
        progress_pct: 0,
        current_file: null,
        estimated_remaining_seconds: null,
        elapsed_seconds: 0,
      });
      pollingRef.current = setInterval(() => pollStatus(resp.batch_id), 2000);
    } catch (e: any) {
      const msg = e?.response?.data?.detail || e?.message || "Upload failed";
      setError(msg);
    } finally {
      setUploading(false);
    }
  };

  const handleReset = () => {
    stopPolling();
    setPhase("upload");
    setImageFiles([]);
    setCsvFile(null);
    setBatchId(null);
    setStatus(null);
    setResults(null);
    setError(null);
  };

  const sortedResults = (results?.results || []).slice().sort((a, b) => {
    let cmp = 0;
    const key = sortKey as keyof BatchFileResult;
    const va = a[key];
    const vb = b[key];
    if (typeof va === "string" && typeof vb === "string") {
      cmp = va.localeCompare(vb);
    } else if (typeof va === "number" && typeof vb === "number") {
      cmp = va - vb;
    }
    return sortAsc ? cmp : -cmp;
  });

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
              Batch Analysis
            </h1>
            <p className="mt-1.5 text-sm text-gray-500">
              Upload multiple images for bulk AI-powered disaster damage assessment
            </p>
          </div>
          {phase !== "upload" && (
            <Button variant="outline" size="sm" onClick={handleReset} className="gap-1.5">
              <RefreshCw className="h-4 w-4" />
              New Batch
            </Button>
          )}
        </motion.div>

        {/* Error */}
        <ErrorAlert
          message={error || ""}
          visible={!!error}
          onRetry={phase === "upload" ? handleUpload : undefined}
          onDismiss={() => setError(null)}
        />

        {/* Upload Phase */}
        {phase === "upload" && (
          <UploadSection
            imageFiles={imageFiles}
            setImageFiles={setImageFiles}
            csvFile={csvFile}
            setCsvFile={setCsvFile}
            uploading={uploading}
            onUpload={handleUpload}
            imageInputRef={imageInputRef}
            csvInputRef={csvInputRef}
          />
        )}

        {/* Processing Phase */}
        {phase === "processing" && status && (
          <ProcessingSection status={status} />
        )}

        {/* Results Phase */}
        {phase === "results" && results && (
          <ResultsSection
            results={results}
            sortedResults={sortedResults}
            sortKey={sortKey}
            sortAsc={sortAsc}
            setSortKey={setSortKey}
            setSortAsc={setSortAsc}
            expandedRow={expandedRow}
            setExpandedRow={setExpandedRow}
          />
        )}
      </div>
    </Layout>
  );
}

// ── Upload Section ──────────────────────────────────────────────────

function UploadSection({
  imageFiles, setImageFiles, csvFile, setCsvFile,
  uploading, onUpload, imageInputRef, csvInputRef,
}: {
  imageFiles: File[]; setImageFiles: (f: File[]) => void;
  csvFile: File | null; setCsvFile: (f: File | null) => void;
  uploading: boolean; onUpload: () => void;
  imageInputRef: React.RefObject<HTMLInputElement | null>;
  csvInputRef: React.RefObject<HTMLInputElement | null>;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      <div className="grid gap-6 md:grid-cols-2">
        {/* Image Upload */}
        <Card>
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-2">
              <FileImage className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-secondary">Image Files</h3>
            </div>
            <div
              onClick={() => imageInputRef.current?.click()}
              className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-gray-50 p-8 transition-colors hover:border-primary/50 hover:bg-primary/5"
            >
              <Upload className="mb-2 h-8 w-8 text-gray-300" />
              <p className="text-sm font-medium text-gray-600">
                Click to select images
              </p>
              <p className="text-xs text-gray-400 mt-1">JPG, PNG, WebP, BMP</p>
            </div>
            <input
              ref={imageInputRef}
              type="file"
              multiple
              accept="image/jpeg,image/png,image/webp,image/bmp"
              className="hidden"
              onChange={(e) => {
                const files = Array.from(e.target.files || []);
                setImageFiles(files);
                e.target.value = "";
              }}
            />
            {imageFiles.length > 0 && (
              <div className="max-h-40 space-y-1 overflow-y-auto">
                <p className="text-xs font-medium text-gray-500">
                  {imageFiles.length} file(s) selected
                </p>
                {imageFiles.map((f, i) => (
                  <div key={i} className="flex items-center gap-2 rounded bg-gray-50 px-2.5 py-1.5 text-xs text-gray-600">
                    <FileImage className="h-3.5 w-3.5 text-primary/60" />
                    <span className="flex-1 truncate">{f.name}</span>
                    <span className="text-gray-400">
                      {(f.size / 1024).toFixed(0)} KB
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* CSV Upload */}
        <Card>
          <CardContent className="p-5 space-y-4">
            <div className="flex items-center gap-2">
              <Table className="h-5 w-5 text-primary" />
              <h3 className="font-semibold text-secondary">CSV Report (optional)</h3>
            </div>
            <div
              onClick={() => csvInputRef.current?.click()}
              className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border bg-gray-50 p-8 transition-colors hover:border-primary/50 hover:bg-primary/5"
            >
              <FileText className="mb-2 h-8 w-8 text-gray-300" />
              <p className="text-sm font-medium text-gray-600">
                Click to select CSV
              </p>
              <p className="text-xs text-gray-400 mt-1">Must have a filename column</p>
            </div>
            <input
              ref={csvInputRef}
              type="file"
              accept=".csv"
              className="hidden"
              onChange={(e) => {
                setCsvFile(e.target.files?.[0] || null);
                e.target.value = "";
              }}
            />
            {csvFile && (
              <div className="flex items-center gap-2 rounded bg-green-50 px-2.5 py-1.5 text-xs text-green-700">
                <CheckCircle className="h-3.5 w-3.5" />
                <span className="flex-1 truncate">{csvFile.name}</span>
                <button
                  onClick={() => setCsvFile(null)}
                  className="text-green-500 hover:text-green-700"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
            <p className="text-[11px] text-gray-400 leading-relaxed">
              CSV with <strong>filename</strong> and optional <strong>text</strong> columns.
              Filenames must match uploaded images to pair them with disaster descriptions.
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Upload Button */}
      <div className="flex justify-center">
        <Button
          onClick={onUpload}
          disabled={uploading || imageFiles.length === 0}
          size="lg"
          className="gap-2 px-8"
        >
          {uploading ? (
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
          ) : (
            <Play className="h-5 w-5" />
          )}
          {uploading ? "Uploading..." : "Upload & Start Analysis"}
        </Button>
      </div>
    </motion.div>
  );
}

// ── Processing Section ──────────────────────────────────────────────

function ProcessingSection({ status }: { status: BatchStatusResponse }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="mx-auto max-w-2xl space-y-6"
    >
      <Card>
        <CardContent className="p-6 space-y-5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full bg-blue-100">
              <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
            </div>
            <div>
              <h3 className="font-semibold text-secondary">Processing Batch</h3>
              <p className="text-xs text-gray-500">ID: {status.batch_id}</p>
            </div>
          </div>

          {/* Progress bar */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-secondary">
                {status.progress_pct.toFixed(0)}%
              </span>
              <span className="text-xs text-gray-500">
                {status.completed} / {status.total_files} files
              </span>
            </div>
            <div className="h-2.5 overflow-hidden rounded-full bg-gray-100">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${status.progress_pct}%` }}
                transition={{ duration: 0.3 }}
                className="h-full rounded-full bg-gradient-to-r from-primary to-blue-500"
              />
            </div>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="rounded-lg bg-green-50 p-3 text-center">
              <p className="text-lg font-bold text-green-600">{status.completed}</p>
              <p className="text-[11px] text-green-700">Completed</p>
            </div>
            <div className="rounded-lg bg-red-50 p-3 text-center">
              <p className="text-lg font-bold text-red-600">{status.failed}</p>
              <p className="text-[11px] text-red-700">Failed</p>
            </div>
            <div className="rounded-lg bg-gray-100 p-3 text-center">
              <p className="text-lg font-bold text-gray-600">{status.pending}</p>
              <p className="text-[11px] text-gray-600">Pending</p>
            </div>
          </div>

          {/* Current file & time */}
          <div className="space-y-2 text-sm text-gray-500">
            <div className="flex items-center gap-2">
              <FileImage className="h-4 w-4 text-primary/60" />
              <span className="truncate">
                {status.current_file || "Waiting..."}
              </span>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {status.elapsed_seconds.toFixed(1)}s elapsed
              </span>
              {status.estimated_remaining_seconds !== null && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  ~{status.estimated_remaining_seconds.toFixed(0)}s remaining
                </span>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}

// ── Results Section ─────────────────────────────────────────────────

function ResultsSection({
  results, sortedResults, sortKey, sortAsc, setSortKey, setSortAsc,
  expandedRow, setExpandedRow,
}: {
  results: BatchResultsResponse;
  sortedResults: BatchFileResult[];
  sortKey: string; sortAsc: boolean;
  setSortKey: (k: string) => void; setSortAsc: (a: boolean) => void;
  expandedRow: string | null; setExpandedRow: (v: string | null) => void;
}) {
  const summary = results.summary;

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(true);
    }
  };

  const SortIcon = ({ col }: { col: string }) => {
    if (sortKey !== col) return <ChevronDown className="h-3 w-3 opacity-30" />;
    return sortAsc ? (
      <ChevronUp className="h-3 w-3" />
    ) : (
      <ChevronDown className="h-3 w-3" />
    );
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Summary Cards */}
      {summary && (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <StatCard icon={<CheckCircle className="h-4 w-4" />} label="Successful" value={summary.successful} color="text-green-600" />
          <StatCard icon={<XCircle className="h-4 w-4" />} label="Failed" value={summary.failed} color="text-red-600" />
          <StatCard icon={<AlertTriangle className="h-4 w-4" />} label="High Priority" value={summary.high_priority} color="text-red-600" />
          <StatCard icon={<Cpu className="h-4 w-4" />} label="Avg Confidence" value={`${summary.avg_confidence.toFixed(1)}%`} color="text-primary" />
        </div>
      )}

      {/* Distribution Charts */}
      {summary && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card>
            <CardContent className="p-4 space-y-3">
              <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
                <BarChart3 className="h-3.5 w-3.5" />
                Severity Distribution
              </h4>
              {Object.entries(summary.severity_distribution).map(([sev, count]) => {
                const pct = summary.total_files > 0 ? (count / summary.total_files) * 100 : 0;
                const barColor = severityColors[sev] || "bg-gray-400";
                return (
                  <div key={sev} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">{sev}</span>
                      <span className="font-medium text-secondary">{count} ({pct.toFixed(0)}%)</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-gray-100">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                        className={`h-full rounded-full ${barColor}`}
                      />
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-4 space-y-3">
              <h4 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-gray-400">
                <List className="h-3.5 w-3.5" />
                Priority Distribution
              </h4>
              {Object.entries(summary.priority_distribution).map(([pri, count]) => {
                const pct = summary.total_files > 0 ? (count / summary.total_files) * 100 : 0;
                const barColor = pri === "High" ? "bg-red-500" : pri === "Medium" ? "bg-yellow-500" : "bg-green-500";
                return (
                  <div key={pri} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-600">{pri}</span>
                      <span className="font-medium text-secondary">{count} ({pct.toFixed(0)}%)</span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-gray-100">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${pct}%` }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                        className={`h-full rounded-full ${barColor}`}
                      />
                    </div>
                  </div>
                );
              })}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Processing info */}
      <div className="flex items-center justify-between text-xs text-gray-400">
        <span>Batch ID: {results.batch_id}</span>
        <span>Processed in {results.processing_time_seconds.toFixed(1)}s</span>
      </div>

      {/* Download buttons — centralized report service */}
      <div className="flex flex-wrap gap-3">
        <a
          href={getReportCsvUrl(results.batch_id)}
          download
          className="inline-flex items-center gap-1.5 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-primary/90"
        >
          <Download className="h-4 w-4" />
          Download CSV
        </a>
        <a
          href={getReportBatchUrl(results.batch_id)}
          download
          className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-white px-4 py-2 text-sm font-medium text-secondary transition-colors hover:bg-gray-50"
        >
          <FileText className="h-4 w-4" />
          Download PDF Report
        </a>
      </div>

      {/* Results Table */}
      <Card>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-border bg-gray-50">
                  <Th onClick={() => toggleSort("filename")} active={sortKey === "filename"}>
                    <span className="flex items-center gap-1">
                      Filename <SortIcon col="filename" />
                    </span>
                  </Th>
                  <Th onClick={() => toggleSort("success")} active={sortKey === "success"}>
                    <span className="flex items-center gap-1">
                      Status <SortIcon col="success" />
                    </span>
                  </Th>
                  <Th onClick={() => toggleSort("severity")} active={sortKey === "severity"}>
                    <span className="flex items-center gap-1">
                      Severity <SortIcon col="severity" />
                    </span>
                  </Th>
                  <Th onClick={() => toggleSort("confidence")} active={sortKey === "confidence"}>
                    <span className="flex items-center gap-1">
                      Confidence <SortIcon col="confidence" />
                    </span>
                  </Th>
                  <Th onClick={() => toggleSort("priority")} active={sortKey === "priority"}>
                    <span className="flex items-center gap-1">
                      Priority <SortIcon col="priority" />
                    </span>
                  </Th>
                  <Th onClick={() => toggleSort("processing_time_ms")} active={sortKey === "processing_time_ms"}>
                    <span className="flex items-center gap-1">
                      Time <SortIcon col="processing_time_ms" />
                    </span>
                  </Th>
                  <Th>Details</Th>
                </tr>
              </thead>
              <tbody>
                {sortedResults.map((r) => (
                  <FragmentRow
                    key={r.filename}
                    r={r}
                    expanded={expandedRow === r.filename}
                    onToggle={() => setExpandedRow(expandedRow === r.filename ? null : r.filename)}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {results.results.length === 0 && (
        <div className="flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-border bg-card p-12 text-center">
          <Layers className="h-10 w-10 text-gray-300 mb-3" />
          <p className="text-sm font-medium text-gray-500">No results available</p>
        </div>
      )}
    </motion.div>
  );
}

function Th({
  children, onClick, active, className,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  active?: boolean;
  className?: string;
}) {
  return (
    <th
      onClick={onClick}
      className={`px-4 py-3 text-xs font-semibold uppercase tracking-wider ${
        onClick ? "cursor-pointer hover:text-primary" : ""
      } ${active ? "text-primary" : "text-gray-500"} ${className || ""}`}
    >
      {children}
    </th>
  );
}

function FragmentRow({
  r, expanded, onToggle,
}: {
  r: BatchFileResult;
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <>
      <tr className="border-b border-border transition-colors hover:bg-gray-50/50">
        <td className="max-w-[200px] truncate px-4 py-3 text-xs font-medium text-secondary">
          <span className="flex items-center gap-1.5">
            <FileImage className="h-3.5 w-3.5 flex-shrink-0 text-primary/60" />
            <span className="truncate">{r.filename}</span>
          </span>
        </td>
        <td className="px-4 py-3">
          {r.success ? (
            <Badge className="bg-green-100 text-green-700 border-green-200 text-[10px]">
              <CheckCircle className="mr-0.5 h-3 w-3" /> OK
            </Badge>
          ) : (
            <Badge className="bg-red-100 text-red-700 border-red-200 text-[10px]">
              <XCircle className="mr-0.5 h-3 w-3" /> FAIL
            </Badge>
          )}
        </td>
        <td className="px-4 py-3 text-xs text-gray-600">
          {r.success ? (r.severity || "-") : "-"}
        </td>
        <td className="px-4 py-3 text-xs font-medium text-secondary">
          {r.success && r.confidence !== null ? `${r.confidence.toFixed(1)}%` : "-"}
        </td>
        <td className="px-4 py-3">
          {r.success && r.priority ? (
            <Badge className={`${priorityColors[r.priority] || ""} text-[10px]`}>
              {r.priority}
            </Badge>
          ) : (
            <span className="text-xs text-gray-400">-</span>
          )}
        </td>
        <td className="px-4 py-3 text-xs text-gray-500">
          {formatMs(r.processing_time_ms)}
        </td>
        <td className="px-4 py-3">
          <button
            onClick={onToggle}
            className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-primary transition-colors"
          >
            {expanded ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>
        </td>
      </tr>
      <AnimatePresence>
        {expanded && (
          <motion.tr
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <td colSpan={7} className="border-b border-border bg-gray-50/70 px-4 py-3">
              <ExpandedDetails r={r} />
            </td>
          </motion.tr>
        )}
      </AnimatePresence>
    </>
  );
}

function ExpandedDetails({ r }: { r: BatchFileResult }) {
  if (!r.success) {
    return (
      <div className="flex items-start gap-2 text-sm text-red-600">
        <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0" />
        <span>{r.error || "Analysis failed with unknown error"}</span>
      </div>
    );
  }
  return (
    <div className="grid gap-x-6 gap-y-2 text-xs sm:grid-cols-2">
      <div>
        <span className="font-medium text-gray-500">Infrastructure Impact:</span>
        <span className="ml-2 text-gray-700">{r.infrastructure_impact || "-"}</span>
      </div>
      <div>
        <span className="font-medium text-gray-500">Emergency Level:</span>
        <span className="ml-2 text-gray-700">{r.emergency_level || "-"}</span>
      </div>
      {r.recommended_actions.length > 0 && (
        <div className="sm:col-span-2 mt-1">
          <span className="font-medium text-gray-500">Actions: </span>
          <span className="text-gray-600">{r.recommended_actions.join("; ")}</span>
        </div>
      )}
    </div>
  );
}

function StatCard({
  icon, label, value, color,
}: {
  icon: React.ReactNode; label: string; value: string | number; color: string;
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-3 p-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/5">
          <span className={color}>{icon}</span>
        </div>
        <div>
          <p className="text-2xl font-bold text-secondary">{value}</p>
          <p className="text-xs text-gray-500">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}
