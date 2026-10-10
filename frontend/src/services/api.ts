import axios from "axios";
import type {
  PredictResponse,
  HealthResponse,
  DisasterEventSummary,
  DisasterEventDetail,
  DisasterAnalysisResponse,
  BatchUploadResponse,
  BatchStatusResponse,
  BatchResultsResponse,
  DashboardSummary,
  DashboardStatistics,
  DashboardChartData,
  TimelineEntry,
  GradCAMResponse,
  AnalysisCenterRecord,
  AnalysisCenterResponse,
  AnalysisCenterStatistics,
} from "../types";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 120000,
});

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

export async function predict(
  image: File,
  tweetText: string
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("image", image);
  form.append("tweet_text", tweetText);
  const { data } = await api.post<PredictResponse>("/predict", form);
  return data;
}

export async function fetchDisasterEvents(
  limit = 20
): Promise<DisasterEventSummary[]> {
  const { data } = await api.get<DisasterEventSummary[]>("/disasters/events", {
    params: { limit },
  });
  return data;
}

export async function fetchDisasterEvent(
  eventId: string
): Promise<DisasterEventDetail> {
  const { data } = await api.get<DisasterEventDetail>(
    `/disasters/events/${eventId}`
  );
  return data;
}

export async function analyzeDisasterEvent(
  eventId: string
): Promise<DisasterAnalysisResponse> {
  const { data } = await api.post<DisasterAnalysisResponse>(
    `/disasters/events/${eventId}/analyze`
  );
  return data;
}

// ── Batch processing ─────────────────────────────────────────────────

export async function uploadBatch(
  files: File[],
  csvFile: File | null
): Promise<BatchUploadResponse> {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  if (csvFile) form.append("csv_file", csvFile);
  const { data } = await api.post<BatchUploadResponse>("/batch/upload", form);
  return data;
}

export async function startBatch(batchId: string): Promise<{ status: string }> {
  const { data } = await api.post<{ status: string }>(
    `/batch/start/${batchId}`
  );
  return data;
}

export async function getBatchStatus(
  batchId: string
): Promise<BatchStatusResponse> {
  const { data } = await api.get<BatchStatusResponse>(
    `/batch/status/${batchId}`
  );
  return data;
}

export async function getBatchResults(
  batchId: string
): Promise<BatchResultsResponse> {
  const { data } = await api.get<BatchResultsResponse>(
    `/batch/results/${batchId}`
  );
  return data;
}

export function getBatchDownloadCsvUrl(batchId: string): string {
  return `${api.defaults.baseURL}/batch/download/csv/${batchId}`;
}

export function getBatchDownloadPdfUrl(batchId: string): string {
  return `${api.defaults.baseURL}/batch/download/pdf/${batchId}`;
}

// ── Explainability ──────────────────────────────────────────────────

export async function generateGradCAM(
  image: File,
  tweetText?: string,
  analysisId?: string
): Promise<GradCAMResponse> {
  const form = new FormData();
  form.append("image", image);
  if (tweetText) form.append("tweet_text", tweetText);
  if (analysisId) form.append("analysis_id", analysisId);
  const { data } = await api.post<GradCAMResponse>("/explainability/gradcam", form);
  return data;
}

// ── Reports ─────────────────────────────────────────────────────────

export function getReportManualUrl(analysisId: string): string {
  return `${api.defaults.baseURL}/report/manual/${analysisId}`;
}

export function getReportLiveUrl(analysisId: string): string {
  return `${api.defaults.baseURL}/report/live/${analysisId}`;
}

export function getReportBatchUrl(batchId: string): string {
  return `${api.defaults.baseURL}/report/batch/${batchId}`;
}

export function getReportCsvUrl(batchId: string): string {
  return `${api.defaults.baseURL}/report/csv/${batchId}`;
}

// ── Dashboard ───────────────────────────────────────────────────────

export async function fetchDashboardSummary(): Promise<DashboardSummary> {
  const { data } = await api.get<DashboardSummary>("/dashboard/summary");
  return data;
}

export async function fetchDashboardStatistics(): Promise<DashboardStatistics> {
  const { data } = await api.get<DashboardStatistics>("/dashboard/statistics");
  return data;
}

export async function fetchDashboardCharts(): Promise<DashboardChartData> {
  const { data } = await api.get<DashboardChartData>("/dashboard/charts");
  return data;
}

// ── Analysis Center ────────────────────────────────────────────────

export async function fetchAnalysisCenter(params: {
  page?: number;
  per_page?: number;
  sort_by?: string;
  query?: string;
  record_type?: string;
  analysis_mode?: string;
  severity?: string;
  priority?: string;
  disaster_type?: string;
  source?: string;
  date_from?: string;
  date_to?: string;
} = {}): Promise<AnalysisCenterResponse> {
  const { data } = await api.get<AnalysisCenterResponse>("/analysis-center", { params });
  return data;
}

export async function fetchAnalysisCenterItem(analysisId: string): Promise<AnalysisCenterRecord> {
  const { data } = await api.get<AnalysisCenterRecord>(`/analysis-center/${analysisId}`);
  return data;
}

export async function searchAnalysisCenter(
  q: string,
  page = 1,
  per_page = 20
): Promise<AnalysisCenterResponse> {
  const { data } = await api.get<AnalysisCenterResponse>("/analysis-center/search", {
    params: { q, page, per_page },
  });
  return data;
}

export function getAnalysisCenterExportUrl(params: Record<string, string> = {}): string {
  const qs = new URLSearchParams(params).toString();
  return `${api.defaults.baseURL}/analysis-center/export${qs ? `?${qs}` : ""}`;
}

export async function fetchAnalysisCenterStatistics(): Promise<AnalysisCenterStatistics> {
  const { data } = await api.get<AnalysisCenterStatistics>("/analysis-center/statistics");
  return data;
}

export async function fetchDashboardTimeline(
  limit = 20
): Promise<TimelineEntry[]> {
  const { data } = await api.get<TimelineEntry[]>("/dashboard/timeline", {
    params: { limit },
  });
  return data;
}
