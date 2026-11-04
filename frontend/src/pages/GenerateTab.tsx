import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { detectAudio, type DetectionResult } from '../services/detectionService'
import { apiFetch, ApiError } from '../config/apiConfig'

type TestSample = {
  id: string
  label: 'real' | 'fake'
  attack: string
}

export default function GenerateTab() {
  const [samples, setSamples] = useState<TestSample[]>([])
  const [loading, setLoading] = useState(true)
  const [testing, setTesting] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, DetectionResult>>({})
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    apiFetch('/test-samples')
      .then(async (res) => {
        if (res.ok) setSamples(await res.json())
        else setError('Could not load test samples — make sure the ASVspoof eval dataset is downloaded')
      })
      .catch(() => setError('Backend unreachable'))
      .finally(() => setLoading(false))
  }, [])

  const testSample = async (sample: TestSample) => {
    setTesting(sample.id)
    setError(null)
    try {
      const res = await apiFetch(`/test-samples/${sample.id}`)
      if (!res.ok) throw new Error('Download failed')
      const blob = await res.blob()
      const file = new File([blob], `${sample.id}.flac`, { type: 'audio/flac' })
      const result = await detectAudio(file, false, 'xlsr_mamba')
      setResults((prev) => ({ ...prev, [sample.id]: result }))
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Detection failed')
    } finally {
      setTesting(null)
    }
  }

  const testAll = async () => {
    for (const s of samples) {
      if (results[s.id]) continue
      await testSample(s)
    }
  }

  const testedCount = Object.keys(results).length
  const correctCount = Object.entries(results).filter(([id, r]) => {
    const sample = samples.find((s) => s.id === id)
    return sample && sample.label === r.label
  }).length

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}>
        <h2 className="text-xl font-semibold text-white mb-1">Test with Eval Samples</h2>
        <p className="text-sm text-gray-400">
          Test against ASVspoof 2019 LA evaluation set — NOT used during training.
        </p>
      </motion.div>

      {loading && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass rounded-xl p-6 text-gray-400 text-sm flex items-center gap-3"
        >
          <svg className="animate-spin h-4 w-4 text-teal-400" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Loading samples...
        </motion.div>
      )}

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-yellow-900/20 border border-yellow-700/50 rounded-xl p-4 text-yellow-300 text-sm"
          >
            {error}
            {samples.length === 0 && (
              <p className="mt-2 text-xs text-yellow-400/70">
                Download the ASVspoof 2019 LA eval set to use this feature.
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {samples.length > 0 && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
          <div className="flex items-center gap-4">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={testAll}
              disabled={testing !== null || testedCount === samples.length}
              className="bg-gradient-to-r from-teal-600 to-cyan-600 hover:from-teal-500 hover:to-cyan-500 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-medium py-2.5 px-6 rounded-xl transition-all text-sm"
            >
              {testing ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-3.5 w-3.5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Testing...
                </span>
              ) : testedCount === samples.length ? 'All tested' : 'Test all samples'}
            </motion.button>
            {testedCount > 0 && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2"
              >
                <div className={`w-2.5 h-2.5 rounded-full ${correctCount === testedCount ? 'bg-green-400' : 'bg-yellow-400'}`} />
                <span className="text-sm text-gray-400">
                  Accuracy: <span className="text-white font-mono">{correctCount}/{testedCount}</span>
                  {' '}({Math.round((correctCount / testedCount) * 100)}%)
                </span>
              </motion.div>
            )}
          </div>

          <div className="overflow-x-auto rounded-xl border border-gray-700/50 glow-border">
            <table className="w-full text-sm">
              <thead className="bg-gray-800/80 text-gray-400">
                <tr>
                  {['Sample ID', 'True Label', 'Attack', 'Predicted', 'Conf.', 'Action'].map((h) => (
                    <th key={h} className="px-4 py-3 text-left font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {samples.map((s, i) => {
                  const r = results[s.id]
                  const correct = r && r.label === s.label
                  return (
                    <motion.tr
                      key={s.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.03 }}
                      className={`transition-colors ${r ? (correct ? 'bg-green-950/10' : 'bg-red-950/10') : 'bg-gray-900/40 hover:bg-gray-800/40'}`}
                    >
                      <td className="px-4 py-3 font-mono text-gray-300 text-xs">{s.id}</td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center gap-1.5 text-xs font-medium uppercase px-2 py-0.5 rounded-full ${
                          s.label === 'fake' ? 'bg-red-900/30 text-red-400' : 'bg-green-900/30 text-green-400'
                        }`}>
                          {s.label}
                        </span>
                      </td>
                      <td className="px-4 py-3 font-mono text-gray-500 text-xs">{s.attack}</td>
                      <td className="px-4 py-3">
                        {r ? (
                          <span className={`inline-flex items-center gap-1 text-xs font-medium uppercase ${r.label === 'fake' ? 'text-red-400' : 'text-green-400'}`}>
                            {r.label} <span className={correct ? 'text-green-400' : 'text-red-400'}>{correct ? '✓' : '✗'}</span>
                          </span>
                        ) : (
                          <span className="text-gray-600 text-xs">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 font-mono text-gray-300 text-xs">
                        {r ? `${Math.round(r.confidence * 100)}%` : '—'}
                      </td>
                      <td className="px-4 py-3">
                        <button
                          onClick={() => testSample(s)}
                          disabled={testing !== null}
                          className="text-xs text-teal-400 hover:text-teal-300 disabled:text-gray-600 transition-colors"
                        >
                          {testing === s.id ? (
                            <svg className="animate-spin h-3.5 w-3.5 inline" viewBox="0 0 24 24">
                              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                          ) : r ? 'Retest' : 'Test'}
                        </button>
                      </td>
                    </motion.tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  )
}
