import { useState } from 'react'
import { createPortal } from 'react-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { hasToken, clearToken, ApiError } from '../config/apiConfig'
import { login, signup } from '../services/authService'

export default function LoginControl() {
  const [open, setOpen] = useState(false)
  const [isSignup, setIsSignup] = useState(false)
  const [loggedIn, setLoggedIn] = useState(hasToken())
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [msg, setMsg] = useState('')

  const handleSubmit = async () => {
    setMsg('')
    try {
      if (isSignup) {
        await signup(username, password)
      } else {
        await login(username, password)
      }
      setLoggedIn(true)
      setPassword('')
      setOpen(false)
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : 'Request failed (network error)')
    }
  }

  const handleLogout = () => {
    clearToken()
    setLoggedIn(false)
  }

  if (loggedIn) {
    return (
      <button
        type="button"
        onClick={handleLogout}
        className="rounded-xl border border-slate-600/50 px-3 py-1.5 text-xs font-medium text-slate-300 transition hover:bg-slate-700/50"
        style={{ background: 'var(--bg-card)' }}
      >
        Log out
      </button>
    )
  }

  return (
    <>
      <button
        type="button"
        onClick={() => { setMsg(''); setIsSignup(false); setOpen(true) }}
        className="rounded-xl bg-gradient-to-r from-teal-600 to-cyan-600 px-4 py-1.5 text-xs font-medium text-white transition hover:from-teal-500 hover:to-cyan-500"
      >
        Log in
      </button>

      {createPortal(
        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-[100] flex items-center justify-center px-4"
              style={{ background: 'rgba(0, 0, 0, 0.7)', backdropFilter: 'blur(4px)' }}
              onClick={() => setOpen(false)}
            >
              <motion.div
                initial={{ scale: 0.9, opacity: 0, y: 20 }}
                animate={{ scale: 1, opacity: 1, y: 0 }}
                exit={{ scale: 0.95, opacity: 0, y: 10 }}
                transition={{ type: 'spring', bounce: 0.25 }}
                className="w-full max-w-sm rounded-2xl p-6 shadow-2xl"
                style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                onClick={(e) => e.stopPropagation()}
              >
              <div className="mb-5 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">
                  {isSignup ? 'Create account' : 'Welcome back'}
                </h3>
                <button
                  type="button"
                  aria-label="Close"
                  onClick={() => setOpen(false)}
                  className="text-slate-500 transition hover:text-white text-lg"
                >
                  ✕
                </button>
              </div>

              <label className="mb-1.5 block text-sm" style={{ color: 'var(--text-secondary)' }}>Username</label>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                placeholder={isSignup ? 'Choose a username' : ''}
                className="mb-3 w-full rounded-xl p-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/50 transition"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}
              />

              <label className="mb-1.5 block text-sm" style={{ color: 'var(--text-secondary)' }}>Password</label>
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
                autoComplete={isSignup ? 'new-password' : 'current-password'}
                placeholder={isSignup ? 'Choose a password' : ''}
                className="mb-5 w-full rounded-xl p-3 text-sm text-slate-100 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-teal-500/50 transition"
                style={{ background: 'var(--bg-primary)', border: '1px solid var(--border)' }}
              />

              <button
                type="button"
                onClick={handleSubmit}
                disabled={!username || !password}
                className="w-full rounded-xl bg-gradient-to-r from-teal-600 to-cyan-600 py-3 text-sm font-semibold text-white transition hover:from-teal-500 hover:to-cyan-500 disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {isSignup ? 'Sign up' : 'Log in'}
              </button>

              <p className="mt-4 text-center text-xs" style={{ color: 'var(--text-muted)' }}>
                {isSignup ? (
                  <>Already have an account?{' '}
                    <button type="button" onClick={() => { setIsSignup(false); setMsg('') }} className="text-teal-400 hover:underline">
                      Log in
                    </button>
                  </>
                ) : (
                  <>Don't have an account?{' '}
                    <button type="button" onClick={() => { setIsSignup(true); setMsg('') }} className="text-teal-400 hover:underline">
                      Sign up
                    </button>
                  </>
                )}
              </p>

              <AnimatePresence>
                {msg && (
                  <motion.p
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-3 text-xs text-red-400 text-center"
                  >
                    {msg}
                  </motion.p>
                )}
              </AnimatePresence>
            </motion.div>
          </motion.div>
          )}
        </AnimatePresence>,
        document.body
      )}
    </>
  )
}
