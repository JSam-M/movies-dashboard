'use client'

import { useState, useEffect, useRef } from 'react'

export default function ScrollJump() {
  const [visible,   setVisible]   = useState(false)
  const [goingDown, setGoingDown] = useState(true)
  const lastY    = useRef(0)
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    const onScroll = () => {
      const y = window.scrollY
      setVisible(y > 200)

      if (y !== lastY.current) {
        const down = y > lastY.current
        lastY.current = y

        if (debounce.current) clearTimeout(debounce.current)
        debounce.current = setTimeout(() => setGoingDown(down), 150)
      }
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => {
      window.removeEventListener('scroll', onScroll)
      if (debounce.current) clearTimeout(debounce.current)
    }
  }, [])

  const jump = () => {
    if (goingDown) window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
    else window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (!visible) return null

  return (
    <button
      onClick={jump}
      className="fixed bottom-8 left-1/2 -translate-x-1/2 z-50 transition-all hover:scale-[1.04]"
      style={{
        width: '36px', height: '36px', borderRadius: '9999px',
        background: 'var(--modal-bg)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--sub)', fontSize: '0.85rem',
        cursor: 'pointer',
      }}
      title={goingDown ? 'Jump to bottom' : 'Back to top'}
    >
      {goingDown ? '↓' : '↑'}
    </button>
  )
}
