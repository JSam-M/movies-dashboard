'use client'

import { useEffect } from 'react'

interface Props { onClose: () => void }

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
        className="relative w-full animate-fade-up"
        style={{
          maxWidth: '520px',
          background: 'rgba(255,255,255,0.97)',
          borderRadius: '28px',
          padding: '48px 44px',
          boxShadow: '0 32px 80px rgba(0,0,0,0.16), 0 8px 24px rgba(0,0,0,0.06)',
          border: '1px solid rgba(255,255,255,0.95)',
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

        {/* Logo mark */}
        <div
          className="mb-8"
          style={{
            width: '36px', height: '36px', borderRadius: '8px',
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
          style={{ fontSize: '0.62rem', fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase', color: 'var(--sub)' }}
        >
          Personal Archive · Since 2019
        </p>

        <h2 className="font-display text-[2rem] font-light text-[var(--text)] leading-tight mt-2 mb-6">
          Why this exists
        </h2>

        <div className="space-y-4">
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            I watch a lot of films. When people find out, the first thing they ask is — <em className="text-[var(--text)] not-italic">"can you share your list?"</em> For years, that list lived in a spreadsheet. This is the spreadsheet, made beautiful.
          </p>
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            The second thing people ask is <em className="text-[var(--text)] not-italic">"what should I watch tonight?"</em> The AI on this page knows every film here — it can match your mood, your language, your taste. It only recommends from films I&apos;ve actually watched.
          </p>
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            And the third reason — I was curious about my own habits. The stats page is just for me.
          </p>
        </div>

        <div
          className="mt-8 pt-6"
          style={{ borderTop: '1px solid rgba(0,0,0,0.06)' }}
        >
          <p className="font-body text-[0.7rem] text-[var(--muted)]">
            800+ films · 2019 – present
          </p>
        </div>
      </div>
    </div>
  )
}
