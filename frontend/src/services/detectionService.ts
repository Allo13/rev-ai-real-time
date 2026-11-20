/** Deepfake detection API calls (POST /detect). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type AttributionSegment = {
  start_s: number
  end_s: number
  importance: number
}

export type Explanation = {
  method: string
  baseline: string
  target_class: number
  frame_duration_ms: number
  attribution_frames: number[]
  top_segments: AttributionSegment[]
  narrative?: string | null
}

export type TimelineSegment = {
  start_s: number
  end_s: number
  spoof_prob: number
  bonafide_prob: number
}

export type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
  duration_s?: number
  timeline?: TimelineSegment[]
  explanation?: Explanation | null
}

export type ModelInfo = {
  key: string
  available: boolean
}

/** Human-readable labels for the detector model keys. */
export const MODEL_LABELS: Record<string, string> = {
  baseline: 'XLSR-300M + Fake-Mamba 24L (Baseline)',
  run_a_12: 'XLSR-300M + Fake-Mamba 12L',
  run_b_8: 'XLSR-300M + Fake-Mamba 8L',
  v2_fast: 'wav2vec2-base + AASIST (Fast)',
  v2_accurate: 'XLS-R 300M + AASIST (Accurate)',
  v2_pretrained: 'XLS-R 300M + AASIST Pretrained (Tak et al.)',
}

/** Fetch the detector registry and its per-model checkpoint availability. */
export const getModels = async (): Promise<ModelInfo[]> => {
  const res = await apiFetch('/models')
  if (!res.ok) throw new ApiError(res.status, `Error ${res.status}`)
  const raw = (await res.json()) as Record<string, { available: boolean }>
  return Object.entries(raw).map(([key, v]) => ({ key, available: !!v.available }))
}

/**
 * Upload an audio file for deepfake detection.
 * @param explain request attribution (slower).
 * @param model detector model key (default 'classical').
 * @throws ApiError with the HTTP status (401 unauthenticated, 422 unusable
 *         audio, 503 model unavailable) and the backend's detail message.
 */
export const detectAudio = async (
  file: File,
  explain = false,
  model = 'classical',
): Promise<DetectionResult> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch(
    `/detect?explain=${explain ? 'true' : 'false'}&model=${encodeURIComponent(model)}`,
    {
      method: 'POST',
      body: form,
    },
  )
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<DetectionResult>
}
