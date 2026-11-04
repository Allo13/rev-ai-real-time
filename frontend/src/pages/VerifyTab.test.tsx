import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import VerifyTab from './VerifyTab'

describe('VerifyTab (Record & Detect)', () => {
  it('renders the start recording button', () => {
    render(<VerifyTab />)
    expect(screen.getByRole('button', { name: /start recording/i })).toBeTruthy()
  })
})
