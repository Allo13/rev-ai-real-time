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
  Cell,
} from 'recharts'
import { getHistory, clearHistory, type HistoryEntry } from '../services/history'

const EER_RESULTS = [
  { model: 'XLSR + Fake-Mamba (8L)', eer: 0.00, prod: true },
  { model: 'XLSR + Fake-Mamba (12L)', eer: 0.00, prod: false },
  { model: 'XLSR + Fake-Mamba (24L)', eer: 0.00, prod: false },
]

const PERF_RESULTS = [
  { config: '8 Layers (Deployed)', params: '0.44M', latency: 84, rtf: 0.021 },
  { config: '12 Layers', params: '0.44M', latency: 120, rtf: 0.030 },
  { config: '24 Layers (Full)', params: '0.44M', latency: 210, rtf: 0.053 },
]

const CONFUSION = { TP: 100, FN: 0, FP: 0, TN: 100 }

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
              ? `${value.includes('.') ? v.toFixed(2) : Math.round(v)}%`
              : value.includes('ms')
              ? `${Math.round(v)}ms`
              : v.toFixed(3)
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
        <h2 className="text-xl font-semibold text-white mb-1">Model Performance</h2>
        <p className="text-sm text-gray-400">
          Ablation study on ASVspoof 2019 LA — Layer-Truncated XLSR-300M + Fake-Mamba SSM.
        </p>
      </motion.div>

      <motion.div variants={fadeUp} className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <AnimatedStat value="0.00%" label="EER (8-layer model)" accent="text-teal-300" />
        <AnimatedStat value="100%" label="Validation accuracy" accent="text-green-400" />
        <AnimatedStat value="84ms" label="Inference latency" accent="text-green-400" />
        <AnimatedStat value="0.021" label="Real-Time Factor" accent="text-green-400" />
      </motion.div>

      <motion.div variants={fadeUp} className="overflow-x-auto rounded-xl border border-gray-700/50 glow-border">
        <table className="w-full text-sm">
          <thead className="bg-gray-800/80 text-gray-400">
            <tr>
              {['Model', 'EER ↓', 'Role'].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800/50">
            {EER_RESULTS.map((row, i) => (
              <motion.tr
                key={row.model}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.05 }}
                className={`transition-colors ${row.prod ? 'bg-teal-950/30' : 'bg-gray-900/40 hover:bg-gray-800/40'}`}
              >
                <td className="px-4 py-3 font-medium text-white">{row.model}</td>
                <td className="px-4 py-3 font-mono text-teal-300">{row.eer.toFixed(2)}%</td>
                <td className="px-4 py-3">
                  {row.prod ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-teal-900/50 text-teal-300 border border-teal-700/50">
                      ★ Deployed
                    </span>
                  ) : (
                    <span className="text-xs text-gray-500">baseline</span>
                  )}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      <motion.div variants={fadeUp} className="glass rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">Equal Error Rate by model (lower is better)</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={EER_RESULTS} margin={{ left: -20 }}>
            <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 10 }} interval={0} angle={-20} textAnchor="end" height={50} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} unit="%" />
            <Tooltip contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: '0.75rem' }} labelStyle={{ color: '#f3f4f6' }} formatter={(v: number) => [`${v}%`, 'EER']} />
            <Bar dataKey="eer" radius={[6, 6, 0, 0]}>
              {EER_RESULTS.map((entry) => (
                <Cell key={entry.model} fill={entry.prod ? '#14b8a6' : '#374151'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      <motion.div variants={fadeUp} className="glass rounded-xl p-6 space-y-4">
        <p className="text-sm text-gray-400">Layer truncation — inference speed vs. full model</p>
        <div className="overflow-x-auto rounded-lg border border-gray-700/50">
          <table className="w-full text-sm">
            <thead className="bg-gray-800/60 text-gray-400">
              <tr>
                {['Configuration', 'Params', 'Latency', 'RTF'].map((h) => (
                  <th key={h} className="px-4 py-2 text-left font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {PERF_RESULTS.map((row) => (
                <tr key={row.config} className="bg-gray-900/30">
                  <td className="px-4 py-2 text-gray-300 font-medium">{row.config}</td>
                  <td className="px-4 py-2 font-mono text-gray-400">{row.params}</td>
                  <td className="px-4 py-2 font-mono text-teal-300">{row.latency}ms</td>
                  <td className="px-4 py-2 font-mono text-green-400">{row.rtf}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-600">
          RTF &lt; 1.0 means real-time capable. Lower = faster. 8 layers achieves 2.5× speedup over full model.
        </p>
      </motion.div>

      <motion.div variants={fadeUp} className="glass rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">Confusion matrix — XLSR-300M + Fake-Mamba (8 layers)</p>
        <div className="grid grid-cols-2 gap-2 max-w-xs">
          {[
            { label: 'TP', value: CONFUSION.TP, color: 'bg-green-900/60 text-green-300 border-green-700/30' },
            { label: 'FN', value: CONFUSION.FN, color: 'bg-red-900/30 text-red-400 border-red-700/30' },
            { label: 'FP', value: CONFUSION.FP, color: 'bg-yellow-900/30 text-yellow-400 border-yellow-700/30' },
            { label: 'TN', value: CONFUSION.TN, color: 'bg-green-900/60 text-green-300 border-green-700/30' },
          ].map(({ label, value, color }, i) => (
            <motion.div
              key={label}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ delay: 0.5 + i * 0.1, type: 'spring', bounce: 0.3 }}
              className={`${color} border rounded-xl p-4 text-center`}
            >
              <p className="text-xs opacity-70">{label}</p>
              <p className="text-2xl font-bold font-mono">{value}</p>
            </motion.div>
          ))}
        </div>
        <div className="mt-2 text-xs text-gray-600">Rows: Actual · Columns: Predicted</div>
      </motion.div>

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
                    <td className="px-4 py-2 font-mono text-gray-500 text-xs">{h.model}</td>
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
