export interface PredictionResult {
  severity: string;
  confidence: number;
}

export interface PredictResponse {
  success: boolean;
  analysis_id?: string;
  prediction: PredictionResult;
  model: string;
  processing_time_ms: number;
}

export interface HealthResponse {
  status: string;
  models_loaded: number;
}

export interface ApiError {
  detail: string;
}

export interface DisasterEventSummary {
  id: string;
  title: string;
  type: string;
  country: string;
  date: string | null;
  description: string;
  source: string;
  analysis_status: string;
}

export interface DisasterEventDetail {
  id: string;
  title: string;
  type: string;
  country: string;
  date: string | null;
  description: string;
  source: string;
  source_url: string;
  latitude: number | null;
  longitude: number | null;
  image_url: string | null;
  severity: number | null;
  analysis_status: string;
  metadata: Record<string, unknown>;
}

export interface DisasterAnalysisResponse {
  success: boolean;
  analysis_id?: string;
  event: {
    id: string;
    title: string;
    type: string;
    country: string;
  };
  analysis: {
    severity: string;
    confidence: number;
    priority: string;
    infrastructure_impact: string;
    emergency_level: string;
    recommended_actions: string[];
  } | null;
  error?: string;
  model_used?: string;
  analysis_mode?: string;
  processing_time_ms: number;
}

export interface BatchUploadResponse {
  batch_id: string;
  total_files: number;
  has_csv: boolean;
  message: string;
}

export interface BatchStatusResponse {
  batch_id: string;
  status: string;
  total_files: number;
  completed: number;
  failed: number;
  pending: number;
  progress_pct: number;
  current_file: string | null;
  estimated_remaining_seconds: number | null;
  elapsed_seconds: number;
}

export interface BatchFileResult {
  filename: string;
  success: boolean;
  severity: string | null;
  confidence: number | null;
  priority: string | null;
  infrastructure_impact: string | null;
  emergency_level: string | null;
  recommended_actions: string[];
  processing_time_ms: number;
  error: string | null;
}

export interface BatchSummary {
  total_files: number;
  successful: number;
  failed: number;
  high_priority: number;
  medium_priority: number;
  low_priority: number;
  avg_confidence: number;
  avg_processing_time_ms: number;
  severity_distribution: Record<string, number>;
  priority_distribution: Record<string, number>;
}

export interface BatchResultsResponse {
  batch_id: string;
  status: string;
  total_files: number;
  processing_time_seconds: number;
  results: BatchFileResult[];
  summary: BatchSummary | null;
}

// ── Dashboard ───────────────────────────────────────────────────────

export interface DashboardSummary {
  total_analyses: number;
  manual: number;
  live: number;
  batch: number;
  high_priority: number;
  average_confidence: number;
  average_processing_time_ms: number;
}

export interface DashboardStatistics {
  total_analyses: number;
  manual: number;
  live: number;
  batch: number;
  high_priority: number;
  medium_priority: number;
  low_priority: number;
  average_confidence: number;
  average_processing_time_ms: number;
  most_common_disaster_type: string;
  most_common_severity: string;
}

export interface DashboardChartData {
  severity: { name: string; value: number }[];
  priority: { name: string; value: number }[];
  disaster_types: { name: string; value: number }[];
  analysis_modes: { name: string; value: number }[];
  confidence: { name: string; value: number }[];
  processing_time: { name: string; value: number }[];
}

// ── Explainability ──────────────────────────────────────────────────

export interface GradCAMResponse {
  success: boolean;
  analysis_id?: string;
  analysis_mode: string;
  prediction: {
    severity: string;
    confidence: number;
  };
  images: {
    original: string;
    heatmap: string;
    overlay: string;
  };
  explanation: {
    technique: string;
    target_layer: string;
    generated_at: string;
  };
}

// ── Analysis Center ────────────────────────────────────────────────

export interface AnalysisCenterRecord {
  analysis_id: string;
  record_type: string;
  timestamp: string;
  source: string;
  severity: string | null;
  confidence: number | null;
  priority: string | null;
  analysis_mode: string | null;
  disaster_type: string | null;
  processing_time_ms: number;
  filename: string | null;
  event_title: string | null;
  batch_id: string | null;
  description: string | null;
  country: string | null;
  model_used: string | null;
  infrastructure_impact: string | null;
  emergency_level: string | null;
  recommended_actions: string[];
  success: boolean;
  error: string | null;
  has_explanation: boolean;
}

export interface AnalysisCenterResponse {
  items: AnalysisCenterRecord[];
  page: number;
  per_page: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface AnalysisCenterStatistics {
  total_analyses: number;
  by_type: Record<string, number>;
  by_severity: Record<string, number>;
  by_priority: Record<string, number>;
  average_confidence: number;
  average_processing_time_ms: number;
}

export interface TimelineEntry {
  id: string;
  type: string;
  timestamp: string;
  severity: string | null;
  confidence: number | null;
  priority: string | null;
  analysis_mode: string | null;
  disaster_type: string | null;
  processing_time_ms: number;
  filename: string | null;
  event_title: string | null;
  batch_id: string | null;
  success: boolean;
  error: string | null;
}
