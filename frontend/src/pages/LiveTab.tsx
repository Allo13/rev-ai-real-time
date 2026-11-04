import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import anime from 'animejs'
import {
  LiveMicStream,
  type StreamEvent,
  type StreamStatus,
} from '../services/streamingService'
import { hasToken } from '../config/apiConfig'

const MAX_TIMELINE = 40

const STATUS_LABEL: Record<StreamStatus, string> = {
  idle: 'Idle',
  connecting: 'Connecting...',
  authenticating: 'Authenticating...',
  live: 'LIVE',
  busy: 'Server busy',
  closed: 'Stream closed',
  error: 'Error',
}

function VerdictCard({ event }: { event: StreamEvent }) {
  const pct = Math.round(event.confidence * 100)
  const fake = event.label === 'fake'
  const numRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (numRef.current) {
      anime({
        targets: numRef.current,
        textContent: [0, pct],
        round: 1,
        duration: 600,
        easing: 'easeOutExpo',
      })
    }
  }, [pct])

  return (
    <motion.div
      key={event.window_id}
      initial={{ scale: 0.9, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      transition={{ type: 'spring', bounce: 0.3 }}
      className={`bklit ${fake ? 'bklit-red' : 'bklit-green'} bklit-active relative flex flex-col items-center gap-2 p-6 rounded-2xl bg-gray-900/80 border-2 ${
        fake ? 'border-red-500/40' : 'border-green-500/40'
      }`}
    >
      <span className="flex items-baseline">
        <span ref={numRef} className={`text-5xl font-bold font-mono tabular-nums ${fake ? 'text-red-400' : 'text-green-400'}`}>0</span>
        <span className={`text-xl font-bold ${fake ? 'text-red-400' : 'text-green-400'}`}>%</span>
      </span>
      <span className={`text-lg font-semibold uppercase tracking-[0.2em] ${fake ? 'text-red-400' : 'text-green-400'}`}>
        {event.label}
      </span>
      <span className="text-xs text-gray-500">window #{event.window_id}</span>
    </motion.div>
  )
}

function Timeline({ events }: { events: StreamEvent[] }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-5 space-y-2"
    >
      <p className="text-sm font-medium text-gray-300">Session verdicts</p>
      <div className="flex items-end gap-px h-16">
        {events.map((e, i) => (
          <motion.div
            key={e.window_id}
            initial={{ height: 0, opacity: 0 }}
            animate={{
              height: `${Math.max(15, e.confidence * 100)}%`,
              opacity: 0.45 + e.confidence * 0.55,
            }}
            transition={{ duration: 0.3, delay: i === events.length - 1 ? 0 : 0 }}
            className={`flex-1 rounded-sm ${e.label === 'fake' ? 'bg-red-500' : 'bg-green-500'}`}
            style={{
              boxShadow: e.label === 'fake'
                ? '0 0 8px rgba(239,68,68,0.3)'
                : '0 0 8px rgba(34,197,94,0.3)',
            }}
            title={`#${e.window_id} ${e.label} ${(e.confidence * 100).toFixed(0)}%`}
          />
        ))}
      </div>
      <p className="text-[11px] text-gray-600">
        Each bar re-scores the session (first at 3s, then every ~2s) — green real, red fake.
      </p>
    </motion.div>
  )
}

function AudioVisualizer({ active }: { active: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!active) return
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf: number
    const bars = 48
    const amplitudes = new Float32Array(bars)

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const w = canvas.width / bars
      for (let i = 0; i < bars; i++) {
        amplitudes[i] += (Math.random() * 0.3 - amplitudes[i] * 0.1)
        amplitudes[i] = Math.max(0.05, Math.min(1, amplitudes[i]))
        const h = amplitudes[i] * canvas.height * 0.8
        const gradient = ctx.createLinearGradient(0, canvas.height - h, 0, canvas.height)
        gradient.addColorStop(0, 'rgba(20, 184, 166, 0.6)')
        gradient.addColorStop(1, 'rgba(20, 184, 166, 0.1)')
        ctx.fillStyle = gradient
        ctx.fillRect(i * w + 1, canvas.height - h, w - 2, h)
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(raf)
  }, [active])

  if (!active) return null

  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: 64 }}
      exit={{ opacity: 0, height: 0 }}
      className="rounded-xl overflow-hidden"
    >
      <canvas ref={canvasRef} width={600} height={64} className="w-full h-16" />
    </motion.div>
  )
}

export default function LiveTab() {
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<LiveMicStream | null>(null)

  useEffect(() => () => streamRef.current?.stop(), [])

  const startStop = async () => {
    if (status === 'live' || status === 'connecting' || status === 'authenticating') {
      streamRef.current?.stop()
      return
    }
    setError(null)
    setEvents([])
    const stream = new LiveMicStream({
      onStatus: setStatus,
      onError: setError,
      onEvent: (e) => setEvents((prev) => [...prev.slice(-(MAX_TIMELINE - 1)), e]),
    })
    streamRef.current = stream
    await stream.start()
  }

  const running = status === 'live' || status === 'connecting' || status === 'authenticating'
  const latest = events.length > 0 ? events[events.length - 1] : null

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
        <h2 className="text-xl font-semibold text-white mb-1">Live Microphone Analysis</h2>
        <p className="text-sm text-gray-400">
          Streams your microphone to the detector in real time — first verdict at 3s, then every ~2s.
        </p>
      </motion.div>

      {!hasToken() && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="bg-yellow-900/20 border border-yellow-700/50 rounded-xl p-4 text-yellow-300 text-sm"
        >
          Log in or sign up first (top-right) — the live stream requires authentication.
        </motion.div>
      )}

      <div className="flex items-center gap-4">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={startStop}
          disabled={!hasToken()}
          className={`flex-1 font-medium py-3 px-6 rounded-xl transition-all text-white disabled:bg-gray-700 disabled:cursor-not-allowed ${
            running
              ? 'bg-gradient-to-r from-red-600 to-rose-600 hover:from-red-500 hover:to-rose-500'
              : 'bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500'
          }`}
        >
          {running ? 'Stop listening' : 'Start live analysis'}
        </motion.button>
        <motion.span
          animate={{
            borderColor: status === 'live' ? 'rgba(34,197,94,0.5)' : 'rgba(55,65,81,1)',
            color: status === 'live' ? '#4ade80' : '#9ca3af',
          }}
          className={`text-sm font-mono px-3 py-2 rounded-xl border bg-gray-800/50`}
        >
          {status === 'live' && <span className="inline-block w-2 h-2 rounded-full bg-green-400 mr-2 pulse-ring" />}
          {STATUS_LABEL[status]}
        </motion.span>
      </div>

      <AnimatePresence>
        <AudioVisualizer active={running} />
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

      {status === 'live' && !latest && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-5 text-sm text-gray-400 flex items-center gap-3"
        >
          <svg className="animate-spin h-4 w-4 text-teal-400" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Listening... first verdict arrives after ~3 seconds of speech.
        </motion.div>
      )}

      <AnimatePresence mode="wait">
        {latest && (
          <div className="space-y-4">
            <VerdictCard event={latest} />
            {latest.final && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="bg-indigo-900/20 border border-indigo-700/50 rounded-xl p-4 text-indigo-300 text-sm"
              >
                Verdict final — stop and restart to analyze a new utterance.
              </motion.div>
            )}
            <Timeline events={events} />
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}
