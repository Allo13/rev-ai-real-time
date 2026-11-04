import { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import anime from 'animejs'
import { detectAudio, type DetectionResult } from '../services/detectionService'
import { ApiError } from '../config/apiConfig'

function ConfidenceResult({ result }: { result: DetectionResult }) {
  const pct = Math.round(result.confidence * 100)
  const fake = result.label === 'fake'
  const numRef = useRef<HTMLSpanElement>(null)
  const ringRef = useRef<SVGCircleElement>(null)

  useEffect(() => {
    if (numRef.current) {
      anime({ targets: numRef.current, textContent: [0, pct], round: 1, duration: 1200, easing: 'easeOutExpo' })
    }
    if (ringRef.current) {
      const circ = 2 * Math.PI * 54
      anime({ targets: ringRef.current, strokeDashoffset: [circ, circ * (1 - result.confidence)], duration: 1500, easing: 'easeOutQuart' })
    }
  }, [pct, result.confidence])

  return (
    <motion.div
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', bounce: 0.3 }}
      className={`bklit ${fake ? 'bklit-red' : 'bklit-green'} bklit-active relative flex flex-col items-center gap-3 p-8 rounded-2xl bg-gray-900/80 border-2 ${fake ? 'border-red-500/40' : 'border-green-500/40'}`}
    >
      <svg width="128" height="128" viewBox="0 0 128 128" className="drop-shadow-lg">
        <circle cx="64" cy="64" r="54" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="8" />
        <circle
          ref={ringRef}
          cx="64" cy="64" r="54" fill="none"
          stroke={fake ? '#ef4444' : '#22c55e'}
          strokeWidth="8" strokeLinecap="round"
          strokeDasharray={2 * Math.PI * 54}
          strokeDashoffset={2 * Math.PI * 54}
          transform="rotate(-90 64 64)"
          style={{ filter: `drop-shadow(0 0 6px ${fake ? 'rgba(239,68,68,0.4)' : 'rgba(34,197,94,0.4)'})` }}
        />
        <text x="64" y="60" textAnchor="middle" fill="white" fontSize="28" fontWeight="bold" fontFamily="monospace">
          <tspan ref={numRef as unknown as React.Ref<SVGTSpanElement>}>0</tspan>
        </text>
        <text x="64" y="80" textAnchor="middle" fill={fake ? '#ef4444' : '#22c55e'} fontSize="11" fontWeight="600" letterSpacing="0.15em">
          {result.label.toUpperCase()}
        </text>
      </svg>
      <div className="grid grid-cols-2 gap-3 w-full max-w-xs">
        {[
          { label: 'Model', value: result.model },
          { label: 'Latency', value: `${result.latency_ms.toFixed(1)}ms` },
        ].map((item) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="glass rounded-lg p-3 text-center"
          >
            <p className="text-[10px] text-gray-500 uppercase tracking-wider">{item.label}</p>
            <p className="text-sm text-white font-mono truncate">{item.value}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

export default function VerifyTab() {
  const [recording, setRecording] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [seconds, setSeconds] = useState(0)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const chunksRef = useRef<Blob[]>([])

  const startRecording = async () => {
    setError(null)
    setResult(null)
    setAudioBlob(null)
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(null)
    setSeconds(0)

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      chunksRef.current = []

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        setAudioBlob(blob)
        setAudioUrl(URL.createObjectURL(blob))
      }

      recorder.start(250)
      mediaRecorderRef.current = recorder
      setRecording(true)

      timerRef.current = setInterval(() => {
        setSeconds((s) => s + 1)
      }, 1000)
    } catch {
      setError('Microphone access denied — allow the mic permission and retry.')
    }
  }

  const stopRecording = () => {
    mediaRecorderRef.current?.stop()
    mediaRecorderRef.current = null
    setRecording(false)
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  const analyzeRecording = async () => {
    if (!audioBlob) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const file = new File([audioBlob], 'recording.webm', { type: 'audio/webm' })
      const data = await detectAudio(file, false, 'xlsr_mamba')
      setResult(data)
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Detection failed')
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, '0')}`

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
        <h2 className="text-xl font-semibold text-white mb-1">Record & Detect</h2>
        <p className="text-sm text-gray-400">
          Record audio from your microphone and run it through the deepfake detector.
        </p>
      </motion.div>

      <div className="flex items-center gap-4">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={recording ? stopRecording : startRecording}
          className={`flex-1 font-medium py-3 px-6 rounded-xl transition-all text-white ${
            recording
              ? 'bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500'
              : 'bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500'
          }`}
        >
          {recording ? `Stop Recording (${formatTime(seconds)})` : 'Start Recording'}
        </motion.button>
      </div>

      <AnimatePresence>
        {recording && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-3 bg-red-900/20 border border-red-800/50 rounded-xl p-4"
          >
            <div className="w-3 h-3 rounded-full bg-red-500 pulse-ring" />
            <span className="text-sm text-red-300">Recording... {formatTime(seconds)}</span>
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {audioUrl && !recording && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="glass rounded-xl p-5 border border-gray-700/50 space-y-4"
          >
            <p className="text-sm text-gray-300 font-medium">Recorded audio ({formatTime(seconds)})</p>
            <audio controls src={audioUrl} className="w-full" />
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={analyzeRecording}
              disabled={loading}
              className="w-full bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-xl transition-all"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Analysing...
                </span>
              ) : 'Detect Deepfake'}
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>

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

      <AnimatePresence>
        {result && <ConfidenceResult result={result} />}
      </AnimatePresence>
    </div>
  )
}
