/** Forensic report API calls (POST /forensic/report). */
import { apiFetch, ApiError } from '../config/apiConfig'
import type { DetectionResult, TimelineSegment } from './detectionService'

export type ForensicReport = {
  report_id: string
  generated_at: string
  analyst: string
  standard: string
  audio_hash_sha256: string
  duration_s: number
  verdict: {
    label: string
    confidence: number
    model: string
    latency_ms: number
  }
  timeline: TimelineSegment[]
  analysis_summary: {
    total_segments_analyzed: number
    peak_spoof_probability: number
    peak_segment: TimelineSegment | null
    avg_spoof_probability: number
    segments_flagged: number
  }
  chain_of_custody: {
    audio_hash_sha256: string
    analysis_timestamp: string
    model_version: string
    report_hash: string
  }
  methodology: string
}

export const generateReport = async (result: DetectionResult): Promise<ForensicReport> => {
  const res = await apiFetch('/forensic/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_hash: result.audio_hash,
      analyst_name: 'Reveal AI Analyst',
    }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<ForensicReport>
}
