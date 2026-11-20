import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import anime from 'animejs'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
} from 'recharts'
import { getHistory, clearHistory, type HistoryEntry } from '../services/history'
import { MODEL_LABELS } from '../services/detectionService'

const MODEL_INFO = {
  name: 'XLS-R 300M + AASIST (Tak et al.)',
  architecture: 'SSL front-end (XLS-R 300M, 24 layers, 1024-dim) + Original AASIST backend',
  dataset: 'ASVspoof 2019 LA',
  source: 'TakHemlata/SSL_Anti-spoofing',
  publishedEER_LA: '0.82%',
  publishedEER_DF: '2.85%',
  sslParams: '315M',
  backendParams: '0.4M',
}

const DATASET_STATS = [
  { split: 'Train', bonafide: 2580, spoofed: 22800, attacks: 'A01–A06' },
  { split: 'Dev', bonafide: 2548, spoofed: 22296, attacks: 'A01–A06' },
  { split: 'Eval', bonafide: 7355, spoofed: 63882, attacks: 'A07–A19' },
]

const DATASET_CHART = DATASET_STATS.map((s) => ({
  split: s.split,
  Bonafide: s.bonafide,
  Spoofed: s.spoofed,
}))

function AnimatedStat({ value, label, accent }: { value: string; label: string; accent: string }) {
  const ref = useRef<HTMLParagraphElement>(null)
  const num = parseFloat(value)

  useEffect(() => {
    if (ref.current && !isNaN(num)) {
      anime({
        targets: ref.current,
        textContent: [0, num],
        round: value.includes('.') ? 100 : 1,
        duration: 1000,
        easing: 'easeOutExpo',
        update: () => {
          if (ref.current) {
            const v = parseFloat(ref.current.textContent || '0')
            ref.current.textContent = value.includes('%')
              ? `${v.toFixed(2)}%`
              : value.includes('M')
              ? `${v.toFixed(1)}M`
              : v.toFixed(2)
          }
        },
        complete: () => {
          if (ref.current) ref.current.textContent = value
        },
      })
    }
  }, [value, num])

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl p-5 bklit bklit-indigo"
    >
      <p ref={ref} className={`text-3xl font-bold font-mono tabular-nums ${accent}`}>{value}</p>
      <p className="text-xs text-gray-400 mt-1">{label}</p>
    </motion.div>
  )
}

const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08 } },
}
const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0 },
}

export default function ResultsTab() {
  const [history, setHistory] = useState<HistoryEntry[]>([])
  useEffect(() => setHistory(getHistory()), [])

  return (
    <motion.div variants={stagger} initial="hidden" animate="show" className="space-y-8">
      <motion.div variants={fadeUp}>
        <h2 className="text-xl font-semibold text-white mb-1">Model & Performance</h2>
        <p className="text-sm text-gray-400">
          Pretrained XLS-R 300M + AASIST — Tak et al. (2022), trained on ASVspoof 2019 LA.
        </p>
      </motion.div>

      {/* Key stats */}
      <motion.div variants={fadeUp} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <AnimatedStat value="0.82%" label="Published EER (LA)" accent="text-teal-300" />
        <AnimatedStat value="2.85%" label="Published EER (DF)" accent="text-cyan-300" />
        <AnimatedStat value="315M" label="SSL front-end params" accent="text-indigo-300" />
        <AnimatedStat value="0.4M" label="AASIST backend params" accent="text-purple-300" />
      </motion.div>

      {/* Model info */}
      <motion.div variants={fadeUp} className="glass rounded-xl p-6 space-y-3">
        <p className="text-sm font-medium text-gray-300">Active Model</p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {[
            { k: 'Architecture', v: MODEL_INFO.architecture },
            { k: 'Training Dataset', v: MODEL_INFO.dataset },
            { k: 'Source', v: MODEL_INFO.source },
            { k: 'SSL Params', v: MODEL_INFO.sslParams },
            { k: 'Backend Params', v: MODEL_INFO.backendParams },
          ].map(({ k, v }) => (
            <div key={k} className="bg-gray-900/50 rounded-lg p-3">
              <p className="text-[10px] text-gray-500 uppercase tracking-wider">{k}</p>
              <p className="text-sm text-gray-200 mt-0.5 font-mono">{v}</p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Dataset breakdown */}
      <motion.div variants={fadeUp} className="glass rounded-xl p-6 space-y-4">
        <p className="text-sm font-medium text-gray-300">ASVspoof 2019 LA Dataset</p>
        <div className="overflow-x-auto rounded-lg border border-gray-700/50">
          <table className="w-full text-sm">
            <thead className="bg-gray-800/60 text-gray-400">
              <tr>
                {['Split', 'Bonafide', 'Spoofed', 'Total', 'Attacks', 'Ratio'].map((h) => (
                  <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {DATASET_STATS.map((row) => (
                <tr key={row.split} className="bg-gray-900/30">
                  <td className="px-4 py-2 text-gray-300 font-medium">{row.split}</td>
                  <td className="px-4 py-2 font-mono text-green-400">{row.bonafide.toLocaleString()}</td>
                  <td className="px-4 py-2 font-mono text-red-400">{row.spoofed.toLocaleString()}</td>
                  <td className="px-4 py-2 font-mono text-gray-300">{(row.bonafide + row.spoofed).toLocaleString()}</td>
                  <td className="px-4 py-2 font-mono text-gray-400">{row.attacks}</td>
                  <td className="px-4 py-2 font-mono text-yellow-400">1:{(row.spoofed / row.bonafide).toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-600">
          Imbalanced ~1:9 ratio. Eval uses 13 unseen attack types (A07–A19) to test generalization.
        </p>
      </motion.div>

      {/* Dataset chart */}
      <motion.div variants={fadeUp} className="glass rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">Dataset split distribution</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={DATASET_CHART} margin={{ left: -20 }}>
            <XAxis dataKey="split" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '0.75rem' }} labelStyle={{ color: '#f3f4f6' }} />
            <Bar dataKey="Bonafide" fill="#22c55e" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Spoofed" fill="#ef4444" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Detection history */}
      <motion.div variants={fadeUp} className="glass rounded-xl p-6">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-gray-400">Your recent detections</p>
          {history.length > 0 && (
            <button
              type="button"
              onClick={() => { clearHistory(); setHistory([]) }}
              className="text-xs text-gray-500 transition hover:text-gray-300"
            >
              Clear
            </button>
          )}
        </div>
        {history.length === 0 ? (
          <p className="text-xs text-gray-600">Run detections in the Detect tab to populate this table.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-gray-700/50">
            <table className="w-full text-sm">
              <thead className="bg-gray-800/60 text-gray-400">
                <tr>
                  {['Time', 'File', 'Verdict', 'Confidence', 'Model'].map((h) => (
                    <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {history.map((h, i) => (
                  <motion.tr
                    key={h.ts}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.03 }}
                    className="bg-gray-900/30"
                  >
                    <td className="px-4 py-2 text-gray-400 whitespace-nowrap">{new Date(h.ts).toLocaleTimeString()}</td>
                    <td className="px-4 py-2 text-gray-300 max-w-[12rem] truncate">{h.filename}</td>
                    <td className="px-4 py-2">
                      <span className={`${h.label === 'fake' ? 'text-red-400' : 'text-green-400'} font-medium uppercase text-xs`}>
                        {h.label}
                      </span>
                    </td>
                    <td className="px-4 py-2 font-mono text-gray-300">{Math.round(h.confidence * 100)}%</td>
                    <td className="px-4 py-2 font-mono text-gray-500 text-xs">{MODEL_LABELS[h.model] ?? h.model}</td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}
