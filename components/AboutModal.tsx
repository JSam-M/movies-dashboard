'use client'

import { useState, useEffect } from 'react'

interface Props { onClose: () => void }

function FeedbackTrigger() {
  const [open, setOpen]     = useState(false)
  const [text, setText]     = useState('')
  const [status, setStatus] = useState<'idle'|'sending'|'done'|'error'>('idle')

  const submit = async () => {
    if (!text.trim() || status === 'sending') return
    setStatus('sending')
    try {
      const res = await fetch('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ body: text.trim() }),
      })
      setStatus(res.ok ? 'done' : 'error')
      if (res.ok) { setText(''); setTimeout(() => setStatus('idle'), 2000) }
      else setTimeout(() => setStatus('idle'), 3000)
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 3000)
    }
  }

  if (!open) return (
    <button
      onClick={() => setOpen(true)}
      className="font-body text-[0.7rem] text-[var(--blue)] hover:opacity-70 transition-opacity"
    >
      Share feedback
    </button>
  )

  return (
    <div className="w-full mt-4">
      {status === 'done' ? (
        <p className="font-body text-[0.7rem] text-[var(--sub)]">Thank you ✓</p>
      ) : (
        <div className="flex gap-2">
          <input
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="Your thoughts…"
            className="flex-1 rounded-xl font-body text-[0.75rem] outline-none"
            style={{ background: 'var(--fill)', border: '1px solid var(--fill-border)', padding: '7px 10px' }}
          />
          <button
            onClick={submit}
            disabled={!text.trim()}
            className="px-3 rounded-xl font-body text-[0.75rem] font-medium transition-all"
            style={{
              background: text.trim() ? '#0071e3' : 'var(--fill-border)',
              color: text.trim() ? 'white' : 'var(--muted)',
            }}
          >
            {status === 'sending' ? '…' : 'Send'}
          </button>
        </div>
      )}
      {status === 'error' && <p className="font-body text-[0.6rem] mt-1" style={{color:'#ff3b30'}}>Something went wrong. Try again.</p>}
    </div>
  )
}

export default function AboutModal({ onClose }: Props) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-[150] flex items-center justify-center p-6 sm:p-8"
      style={{ background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(16px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full animate-fade-up rounded-3xl"
        style={{
          maxWidth: '520px',
          background: 'var(--modal-bg)',
          padding: '48px 44px',
          boxShadow: '0 32px 80px rgba(0,0,0,0.18), 0 8px 24px rgba(0,0,0,0.06)',
          border: '1px solid var(--glass-border)',
        }}
        onClick={e => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--text)] transition-colors p-1"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>

        <div
          className="mb-8"
          style={{
            width: '36px', height: '36px', borderRadius: '12px',
            background: '#0071e3',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'Georgia, serif', fontSize: '16px', fontWeight: 300,
            color: 'white', letterSpacing: '-0.5px',
          }}
        >
          fc
        </div>

        <p
          className="font-body mb-1"
          style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--sub)' }}
        >
          Personal Archive · Since 2019
        </p>

        <h2 className="font-display text-[2rem] font-light text-[var(--text)] leading-tight mt-2 mb-6">
          Why this exists
        </h2>

        <div className="space-y-4">
          <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed">
            I watch a lot of films. When people find out, the first thing they ask is — <em className="text-[var(--text)] not-italic">&ldquo;can you share your list?&rdquo;</em> For years, that list lived in a spreadsheet. This is the spreadsheet, made beautiful.
          </p>
          <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed">
            The second thing people ask is <em className="text-[var(--text)] not-italic">&ldquo;what should I watch tonight?&rdquo;</em> The AI on this page knows every film here — it can match your mood, your language, your taste. It only recommends from films I&apos;ve actually watched.
          </p>
          <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed">
            And the third reason — I was curious about my own habits. The stats page is just for me.
          </p>
        </div>

        <div className="mt-6 pt-5" style={{ borderTop: '1px solid var(--separator)' }}>
          <p
            className="font-body mb-2"
            style={{ fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--sub)' }}
          >
            Privacy
          </p>
          <p className="font-body text-[0.7rem] text-[var(--muted)] leading-relaxed">
            This site keeps lightweight, anonymous visit stats — pages viewed, device type, and country — tied to a
            random ID that resets when you close your browser. No IP addresses are stored and no cookies are used.
            Chat messages are processed by Anthropic&apos;s Claude API to generate recommendations. Feedback you share
            below is posted as a public GitHub issue, so please don&apos;t include personal details.
          </p>
        </div>

        <div
          className="mt-6 pt-5 flex items-center justify-between"
          style={{ borderTop: '1px solid var(--separator)' }}
        >
          <p className="font-body text-[0.7rem] text-[var(--muted)]">800+ films · 2019 – present</p>
          <FeedbackTrigger />
        </div>
      </div>
    </div>
  )
}
