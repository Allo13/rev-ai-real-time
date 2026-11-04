import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import anime from 'animejs'
import {
  detectAudio,
  getModels,
  MODEL_LABELS,
  type DetectionResult,
  type Explanation,
  type ModelInfo,
} from '../services/detectionService'
import { generateReport } from '../services/forensicsService'
import { addHistory } from '../services/history'
import { ApiError } from '../config/apiConfig'

const EMPTY_AUDIO_SEGMENTS: { start_s: number; end_s: number; importance: number }[] = []

function ConfidenceGauge({ confidence, label }: { confidence: number; label: string }) {
  const pct = Math.round(confidence * 100)
  const isFake = label === 'fake'
  const numRef = useRef<HTMLSpanElement>(null)
  const ringRef = useRef<SVGCircleElement>(null)

  useEffect(() => {
    if (numRef.current) {
      anime({
        targets: numRef.current,
        textContent: [0, pct],
        round: 1,
        duration: 1200,
        easing: 'easeOutExpo',
      })
    }
    if (ringRef.current) {
      const circumference = 2 * Math.PI * 54
      anime({
        targets: ringRef.current,
        strokeDashoffset: [circumference, circumference * (1 - confidence)],
        duration: 1500,
        easing: 'easeOutExpo',
      })
    }
  }, [pct, confidence])

  return (
    <motion.div
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', bounce: 0.4, duration: 0.8 }}
      className={`bklit ${isFake ? 'bklit-red' : 'bklit-green'} bklit-active relative flex flex-col items-center gap-3 p-8 rounded-2xl bg-gray-900/80 border ${
        isFake ? 'border-red-500/40' : 'border-green-500/40'
      }`}
    >
      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" stroke="#1f2937" strokeWidth="6" fill="none" />
          <circle
            ref={ringRef}
            cx="60" cy="60" r="54"
            stroke={isFake ? '#ef4444' : '#22c55e'}
            strokeWidth="6" fill="none"
            strokeLinecap="round"
            strokeDasharray={2 * Math.PI * 54}
            strokeDashoffset={2 * Math.PI * 54}
            style={{ filter: `drop-shadow(0 0 8px ${isFake ? 'rgba(239,68,68,0.5)' : 'rgba(34,197,94,0.5)'})` }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="flex items-baseline">
            <span ref={numRef} className={`text-4xl font-bold font-mono tabular-nums ${isFake ? 'text-red-400' : 'text-green-400'}`}>0</span>
            <span className={`text-lg font-bold ${isFake ? 'text-red-400' : 'text-green-400'}`}>%</span>
          </span>
        </div>
      </div>
      <span className={`text-lg font-semibold uppercase tracking-[0.2em] ${isFake ? 'text-red-400' : 'text-green-400'}`}>
        {label}
      </span>
      <span className="text-xs text-gray-500">confidence</span>
    </motion.div>
  )
}

function AudioWaveform({
  file,
  segments = EMPTY_AUDIO_SEGMENTS,
}: {
  file: File
  segments?: { start_s: number; end_s: number; importance: number }[]
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    let audioContext: AudioContext | null = null

    const drawWaveform = async () => {
      try {
        audioContext = new AudioContext()
        const buffer = await file.arrayBuffer()
        const audio = await audioContext.decodeAudioData(buffer)
        if (cancelled || !canvasRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const data = audio.getChannelData(0)
        const width = canvas.width
        const height = canvas.height
        const step = Math.ceil(data.length / width)

        ctx.clearRect(0, 0, width, height)
        ctx.fillStyle = 'rgba(239, 68, 68, 0.15)'
        for (const segment of segments) {
          const startX = Math.max(0, (segment.start_s / audio.duration) * width)
          const endX = Math.min(width, (segment.end_s / audio.duration) * width)
          if (endX > startX) ctx.fillRect(startX, 0, endX - startX, height)
        }

        const gradient = ctx.createLinearGradient(0, 0, width, 0)
        gradient.addColorStop(0, '#06b6d4')
        gradient.addColorStop(0.5, '#6366f1')
        gradient.addColorStop(1, '#a855f7')

        for (let x = 0; x < width; x++) {
          let peak = 0
          for (let i = 0; i < step; i++) {
            peak = Math.max(peak, Math.abs(data[x * step + i] ?? 0))
          }
          const barHeight = Math.max(2, peak * height)
          ctx.fillStyle = gradient
          ctx.globalAlpha = 0.4 + peak * 0.6
          ctx.fillRect(x, (height - barHeight) / 2, 1, barHeight)
        }
        ctx.globalAlpha = 1
      } catch (error) {
        console.error('Waveform generation failed:', error)
      } finally {
        if (audioContext && audioContext.state !== 'closed') await audioContext.close()
      }
    }
    drawWaveform()
    return () => {
      cancelled = true
      if (audioContext && audioContext.state !== 'closed') void audioContext.close()
    }
  }, [file, segments])

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="bklit bklit-cyan bg-gray-900/80 border border-cyan-500/20 rounded-xl p-4 relative overflow-hidden scan-line"
    >
      <p className="text-sm text-cyan-300 mb-3 font-medium">Audio Waveform</p>
      <canvas ref={canvasRef} width={1000} height={160} className="w-full h-32" />
    </motion.div>
  )
}

function AudioSpectrogram({ file }: { file: File }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    let cancelled = false
    let audioContext: AudioContext | null = null

    const draw = async () => {
      try {
        audioContext = new AudioContext()
        const buffer = await file.arrayBuffer()
        const audio = await audioContext.decodeAudioData(buffer)
        if (cancelled || !canvasRef.current) return

        const data = audio.getChannelData(0)
        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        if (!ctx) return

        const width = canvas.width
        const height = canvas.height
        const size = 1024
        const hop = 512
        const half = size / 2

        const real = new Float64Array(size)
        const imag = new Float64Array(size)
        const hann = new Float64Array(size)
        for (let i = 0; i < size; i++) {
          hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)))
        }

        const image = ctx.createImageData(width, height)
        const frames = Math.max(1, Math.floor((data.length - size) / hop) + 1)

        for (let x = 0; x < width; x++) {
          if (cancelled) return
          const frame = Math.min(frames - 1, Math.floor((x / width) * frames))
          const start = frame * hop
          real.fill(0)
          imag.fill(0)
          for (let i = 0; i < size; i++) real[i] = (data[start + i] ?? 0) * hann[i]

          let j = 0
          for (let i = 1; i < size; i++) {
            let bit = size >> 1
            while (j & bit) { j ^= bit; bit >>= 1 }
            j ^= bit
            if (i < j) [real[i], real[j]] = [real[j], real[i]]
          }

          for (let len = 2; len <= size; len <<= 1) {
            const angle = (-2 * Math.PI) / len
            const wr = Math.cos(angle), wi = Math.sin(angle)
            for (let i = 0; i < size; i += len) {
              let ur = 1, ui = 0
              for (let k = 0; k < len / 2; k++) {
                const even = i + k, odd = even + len / 2
                const tr = ur * real[odd] - ui * imag[odd]
                const ti = ur * imag[odd] + ui * real[odd]
                real[odd] = real[even] - tr
                imag[odd] = imag[even] - ti
                real[even] += tr
                imag[even] += ti
                const nextUr = ur * wr - ui * wi
                ui = ur * wi + ui * wr
                ur = nextUr
              }
            }
          }

          for (let y = 0; y < height; y++) {
            const bin = Math.min(half - 1, Math.max(1, Math.floor((1 - y / height) * half)))
            const magnitude = Math.sqrt(real[bin] ** 2 + imag[bin] ** 2) / size
            const intensity = Math.max(0, Math.min(255, (20 * Math.log10(magnitude + 1e-7) + 100) * 3))
            const p = (y * width + x) * 4
            image.data[p] = Math.min(255, intensity * 0.4)
            image.data[p + 1] = Math.min(255, intensity * 0.2)
            image.data[p + 2] = Math.min(255, 60 + intensity * 0.9)
            image.data[p + 3] = 255
          }
        }
        if (!cancelled) ctx.putImageData(image, 0, 0)
      } catch (error) {
        console.error('Spectrogram generation failed:', error)
      } finally {
        if (audioContext && audioContext.state !== 'closed') await audioContext.close()
      }
    }
    void draw()
    return () => {
      cancelled = true
      if (audioContext && audioContext.state !== 'closed') void audioContext.close()
    }
  }, [file])

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="bg-gray-900/80 border border-purple-500/20 rounded-xl p-4"
    >
      <p className="text-sm text-purple-300 mb-3 font-medium">Audio Spectrogram</p>
      <canvas ref={canvasRef} width={1000} height={240} className="w-full h-48 rounded-lg" />
      <p className="text-xs text-gray-500 mt-2">Frequency intensity over time</p>
    </motion.div>
  )
}

export default function DetectTab() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [explain, setExplain] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('classical')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels()
      .then((ms) => {
        const order = Object.keys(MODEL_LABELS)
        ms.sort(
          (a, b) => Number(b.available) - Number(a.available) || order.indexOf(a.key) - order.indexOf(b.key),
        )
        setModels(ms)
        const first = ms.find((m) => m.available)
        if (first) setModel(first.key)
      })
      .catch(() => setModels([]))
  }, [])

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const data = await detectAudio(file, explain, model)
      setResult(data)
      addHistory(file.name, data)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.status === 401 ? 'Not authenticated — use the Log in button (top-right) first.' : e.message)
      } else {
        setError('Could not reach the detection API. Please try again in a moment.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReport = async () => {
    if (!result) return
    setReportBusy(true)
    try {
      const report = await generateReport(result)
      window.open(report.report_url, '_blank')
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not generate the report.')
    } finally {
      setReportBusy(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.4 }}>
        <h2 className="text-xl font-semibold text-white mb-1">Upload Audio for Analysis</h2>
        <p className="text-sm text-gray-400">
          Supported formats: WAV, MP3, FLAC · Drag & drop or click to select
        </p>
      </motion.div>

      {/* Upload area */}
      <motion.div
        initial={{ opacity: 0, scale: 0.97 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current?.click()}
        className={`relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 ${
          dragOver
            ? 'border-teal-400 bg-teal-500/10 scale-[1.01]'
            : file
            ? 'border-teal-500/40 bg-teal-500/5'
            : 'border-gray-700 hover:border-teal-500/50 hover:bg-gray-900/50'
        }`}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,audio/*"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <motion.div initial={{ scale: 0.9 }} animate={{ scale: 1 }} className="space-y-1">
            <div className="w-12 h-12 mx-auto rounded-xl bg-teal-600/20 border border-teal-500/30 flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 19V6l12-3v13M9 19c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zm12-3c0 1.105-1.343 2-3 2s-3-.895-3-2 1.343-2 3-2 3 .895 3 2zM9 10l12-3" />
              </svg>
            </div>
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-gray-400 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
          </motion.div>
        ) : (
          <div className="space-y-3">
            <div className="w-16 h-16 mx-auto rounded-2xl bg-gray-800/80 border border-gray-700 flex items-center justify-center animate-float">
              <svg className="w-8 h-8 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 16.5V9.75m0 0l3 3m-3-3l-3 3M6.75 19.5a4.5 4.5 0 01-1.41-8.775 5.25 5.25 0 0110.233-2.33 3 3 0 013.758 3.848A3.752 3.752 0 0118 19.5H6.75z" />
              </svg>
            </div>
            <p className="text-gray-300">Click to select or drag & drop audio file</p>
            <p className="text-gray-600 text-sm">WAV, MP3, FLAC · a few seconds of speech</p>
          </div>
        )}
      </motion.div>

      <AnimatePresence>
        {file && (
          <>
            <AudioWaveform file={file} segments={result?.explanation?.top_segments ?? []} />
            <AudioSpectrogram file={file} />
          </>
        )}
      </AnimatePresence>

      {/* Model picker */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }} className="space-y-1">
        <label className="text-sm text-gray-400">Detector model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full bg-gray-800/80 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm focus:border-teal-500 focus:outline-none transition-colors"
        >
          {models.map((m) => (
            <option key={m.key} value={m.key} disabled={!m.available}>
              {(MODEL_LABELS[m.key] ?? m.key) + (m.available ? '' : ' — checkpoint unavailable')}
            </option>
          ))}
        </select>
      </motion.div>

      {/* Options + submit */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-400 select-none cursor-pointer">
          <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)} className="accent-teal-500" />
          Explain decision
        </label>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleUpload}
          disabled={!file || loading}
          className="flex-1 relative overflow-hidden bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-xl transition-all"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Analysing...
            </span>
          ) : (
            'Detect Deepfake'
          )}
        </motion.button>
      </div>

      {/* Error */}
      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-red-900/30 border border-red-700/50 rounded-xl p-4 text-red-300 text-sm"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="space-y-4"
          >
            <ConfidenceGauge confidence={result.confidence} label={result.label} />

            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              {[
                { k: 'Model', v: result.model },
                { k: 'Latency', v: `${result.latency_ms.toFixed(1)}ms` },
                { k: 'Audio Hash', v: result.audio_hash.slice(0, 8) + '...' },
              ].map(({ k, v }, i) => (
                <motion.div
                  key={k}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 * i + 0.3 }}
                  className="glass rounded-xl p-3"
                >
                  <p className="text-xs text-gray-500">{k}</p>
                  <p className="text-sm text-white font-mono truncate">{v}</p>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.6 }}
              className="glass rounded-xl p-5"
            >
              <p className="text-sm font-medium text-gray-300 mb-1">
                {result.label === 'fake'
                  ? 'This audio shows synthetic-speech artifacts.'
                  : 'This audio is consistent with genuine human speech.'}
              </p>
              <p className="text-xs text-gray-500">
                Scored by the {result.model} detector (Layer-Truncated XLSR-300M front-end +
                Fake-Mamba SSM back-end). Uploaded audio is deleted after analysis.
              </p>
            </motion.div>

            {result.explanation && (
              <AttributionView explanation={result.explanation} label={result.label} />
            )}

            <motion.button
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.99 }}
              onClick={handleReport}
              disabled={reportBusy}
              className="w-full border border-gray-700 bg-gray-800/80 hover:bg-gray-700/80 disabled:opacity-50 text-gray-200 font-medium py-2.5 px-6 rounded-xl transition-colors text-sm"
            >
              {reportBusy ? 'Generating...' : 'Download forensic PDF report'}
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function AttributionView({ explanation, label }: { explanation: Explanation; label: string }) {
  const { attribution_frames: frames, top_segments: segments, method, frame_duration_ms } = explanation
  const isFake = label === 'fake'
  const barColor = isFake ? 'bg-rose-500' : 'bg-emerald-500'
  const accentText = isFake ? 'text-rose-300' : 'text-emerald-300'
  const totalS = (frames.length * (frame_duration_ms || 10)) / 1000
  const methodLabel = method === 'integrated_gradients' ? 'Integrated Gradients' : method === 'occlusion' ? 'Occlusion' : method
  const step = Math.max(1, Math.ceil(frames.length / 140))
  const bars: number[] = []
  for (let i = 0; i < frames.length; i += step) bars.push(Math.max(...frames.slice(i, i + step)))
  const top = [...segments].sort((a, b) => b.importance - a.importance).slice(0, 5)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-5 space-y-4"
    >
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <p className="text-sm font-medium text-gray-200">Why this verdict — moment-by-moment</p>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-900 border border-gray-700 text-gray-400">
          {methodLabel} · {frame_duration_ms || 10}ms frames
        </span>
      </div>
      <div>
        <div className="flex items-end gap-px h-20">
          {bars.map((v, i) => (
            <motion.div
              key={i}
              initial={{ height: 0 }}
              animate={{ height: `${Math.max(3, v * 100)}%` }}
              transition={{ delay: i * 0.003, duration: 0.3 }}
              className={`flex-1 rounded-sm ${barColor}`}
              style={{ opacity: 0.35 + v * 0.65 }}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] font-mono text-gray-600 mt-1">
          <span>0.0s</span>
          <span>{(totalS / 2).toFixed(1)}s</span>
          <span>{totalS.toFixed(1)}s</span>
        </div>
      </div>
      {top.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] uppercase tracking-wider text-gray-500">Most influential moments</p>
          {top.map((s, i) => (
            <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 * i }} className="flex items-center gap-3">
              <span className="text-xs font-mono text-gray-500 w-4 text-right">{i + 1}</span>
              <span className="text-xs font-mono text-gray-300 w-24 shrink-0">{s.start_s.toFixed(1)}–{s.end_s.toFixed(1)}s</span>
              <div className="flex-1 h-2 rounded-full bg-gray-900 overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(4, s.importance * 100)}%` }}
                  transition={{ duration: 0.6, delay: 0.1 * i }}
                  className={`h-full ${barColor}`}
                />
              </div>
              <span className={`text-xs font-mono ${accentText} w-10 text-right`}>{Math.round(s.importance * 100)}%</span>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  )
}
