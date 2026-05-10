'use client'

import { useState, useEffect } from 'react'

export default function ScrollJump() {
  const [visible, setVisible]   = useState(false)
  const [atBottom, setAtBottom] = useState(false)

  useEffect(() => {
    const onScroll = () => {
      const scrolled  = window.scrollY
      const maxScroll = document.documentElement.scrollHeight - window.innerHeight
      setVisible(scrolled > 200)
      setAtBottom(maxScroll > 0 && scrolled / maxScroll > 0.4)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const jump = () => {
    if (atBottom) window.scrollTo({ top: 0, behavior: 'smooth' })
    else window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  }

  if (!visible) return null

  return (
    <button
      onClick={jump}
      className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 transition-all hover:scale-[1.04]"
      style={{
        width: '36px', height: '36px', borderRadius: '9999px',
        background: 'rgba(255,255,255,0.85)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--sub)', fontSize: '0.85rem',
        cursor: 'pointer',
      }}
      title={atBottom ? 'Back to top' : 'Jump to bottom'}
    >
      {atBottom ? '↑' : '↓'}
    </button>
  )
}
