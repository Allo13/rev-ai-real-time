import { apiFetch, setToken, ApiError } from '../config/apiConfig'

export const login = async (username: string, password: string): Promise<void> => {
  const body = new URLSearchParams({ username, password })
  const res = await apiFetch('/token', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new ApiError(res.status, data.detail || `Login failed (${res.status})`)
  }
  const data = await res.json()
  if (!data.access_token) throw new ApiError(500, 'No access_token in response')
  setToken(data.access_token)
}

export const signup = async (username: string, password: string): Promise<void> => {
  const body = new URLSearchParams({ username, password })
  const res = await apiFetch('/signup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new ApiError(res.status, data.detail || `Signup failed (${res.status})`)
  }
  const data = await res.json()
  if (!data.access_token) throw new ApiError(500, 'No access_token in response')
  setToken(data.access_token)
}
