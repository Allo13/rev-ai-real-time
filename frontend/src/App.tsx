import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import anime from 'animejs'
import DetectTab from './pages/DetectTab'
import LiveTab from './pages/LiveTab'
import GenerateTab from './pages/GenerateTab'
import VerifyTab from './pages/VerifyTab'
import ResultsTab from './pages/ResultsTab'
import LoginControl from './components/LoginControl'

type Tab = 'detect' | 'live' | 'generate' | 'verify' | 'results'

const pageVariants = {
  enter: { opacity: 0, y: 20, filter: 'blur(8px)' },
  center: { opacity: 1, y: 0, filter: 'blur(0px)' },
  exit: { opacity: 0, y: -10, filter: 'blur(4px)' },
}

function AnimatedLogo() {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current) return
    anime({
      targets: svgRef.current.querySelectorAll('.logo-ring'),
      strokeDashoffset: [anime.setDashoffset, 0],
      duration: 2000,
      easing: 'easeInOutQuart',
    })
    anime({
      targets: svgRef.current.querySelector('.logo-check'),
      strokeDashoffset: [anime.setDashoffset, 0],
      duration: 800,
      delay: 1200,
      easing: 'easeOutQuart',
    })
  }, [])

  return (
    <svg ref={svgRef} width="32" height="32" viewBox="0 0 32 32" fill="none" className="shrink-0">
      <circle className="logo-ring" cx="16" cy="16" r="14" stroke="url(#logo-grad)" strokeWidth="2.5" strokeLinecap="round" />
      <path className="logo-check" d="M10 16 L14.5 20.5 L22 12" stroke="#5eead4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <defs>
        <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32">
          <stop offset="0%" stopColor="#14b8a6" />
          <stop offset="100%" stopColor="#0ea5e9" />
        </linearGradient>
      </defs>
    </svg>
  )
}

function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf: number
    const particles: { x: number; y: number; vx: number; vy: number; r: number; a: number }[] = []

    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener('resize', resize)

    for (let i = 0; i < 40; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.3,
        vy: (Math.random() - 0.5) * 0.3,
        r: Math.random() * 1.5 + 0.5,
        a: Math.random() * 0.3 + 0.1,
      })
    }

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      for (const p of particles) {
        p.x += p.vx
        p.y += p.vy
        if (p.x < 0) p.x = canvas.width
        if (p.x > canvas.width) p.x = 0
        if (p.y < 0) p.y = canvas.height
        if (p.y > canvas.height) p.y = 0
        ctx.beginPath()
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(20, 184, 166, ${p.a})`
        ctx.fill()
      }
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const d = Math.sqrt(dx * dx + dy * dy)
          if (d < 120) {
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(20, 184, 166, ${0.08 * (1 - d / 120)})`
            ctx.stroke()
          }
        }
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none z-0" />
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('detect')

  const tabs: { id: Tab; label: string; icon: string }[] = [
    { id: 'detect', label: 'Detect', icon: 'M9 12l2 2 4-4' },
    { id: 'live', label: 'Live', icon: 'M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m-4-1h8' },
    { id: 'generate', label: 'Test Samples', icon: 'M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2' },
    { id: 'verify', label: 'Record', icon: 'M12 14l9-5-9-5-9 5 9 5zm0 7l9-5-9-5-9 5 9 5z' },
    { id: 'results', label: 'Results', icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' },
  ]

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      <ParticleField />

      <motion.header
        initial={{ y: -60, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-20 glass px-6 py-4"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <AnimatedLogo />
            <span className="text-white font-semibold text-lg tracking-tight">
              Reveal <span className="bg-gradient-to-r from-teal-400 to-cyan-400 bg-clip-text text-transparent">AI</span>
            </span>
            <span className="text-sm hidden sm:block" style={{ color: 'var(--text-muted)' }}>
              Real-Time Voice Deepfake Detection
            </span>
          </div>
          <div className="flex items-center gap-3">
            <LoginControl />
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>v1.0.0</span>
          </div>
        </div>
      </motion.header>

      <motion.nav
        initial={{ y: -40, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-20 glass px-6"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <div className="max-w-5xl mx-auto flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`relative px-5 py-3 text-sm font-medium transition-all duration-300 ${
                activeTab === tab.id
                  ? 'text-teal-300'
                  : 'text-slate-500 hover:text-slate-200'
              }`}
            >
              <span className="flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={tab.icon} />
                </svg>
                {tab.label}
              </span>
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r from-teal-500 to-cyan-500"
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }}
                />
              )}
            </button>
          ))}
        </div>
      </motion.nav>

      <main className="relative z-10 flex-1 px-6 py-8">
        <div className="max-w-5xl mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              variants={pageVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
            >
              {activeTab === 'detect' && <DetectTab />}
              {activeTab === 'live' && <LiveTab />}
              {activeTab === 'generate' && <GenerateTab />}
              {activeTab === 'verify' && <VerifyTab />}
              {activeTab === 'results' && <ResultsTab />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>

      <footer className="relative z-10 glass px-6 py-3" style={{ borderTop: '1px solid var(--border)' }}>
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs" style={{ color: 'var(--text-muted)' }}>
          <span>Canadian University Dubai · GP2 2025/26</span>
          <span>Apache 2.0 License</span>
        </div>
      </footer>
    </div>
  )
}
