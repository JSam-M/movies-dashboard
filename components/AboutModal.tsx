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
      className="fixed inset-0 z-[150] flex items-center justify-center p-6"
      style={{ background: 'rgba(0,0,0,0.2)', backdropFilter: 'blur(16px)' }}
      onClick={onClose}
    >
      <div
        className="relative w-full animate-fade-up"
        style={{
          maxWidth: '520px',
          background: 'rgba(255,255,255,0.97)',
          backdropFilter: 'blur(40px) saturate(1.8)',
          borderRadius: '28px',
          padding: '48px',
          boxShadow: '0 40px 100px rgba(0,0,0,0.18), 0 8px 24px rgba(0,0,0,0.08)',
          border: '1px solid rgba(255,255,255,0.9)',
        }}
        onClick={e => e.stopPropagation()}
      >
        <button
          onClick={onClose}
          className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--text)] transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>

        {/* Logo mark */}
        <div
          className="mb-8"
          style={{
            width: '40px', height: '40px', borderRadius: '10px',
            background: '#0071e3',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontFamily: 'Georgia, serif', fontSize: '18px', fontWeight: 300,
            color: 'white', letterSpacing: '-0.5px',
          }}
        >
          fc
        </div>

        <p className="font-body text-[0.6rem] font-semibold tracking-[0.18em] uppercase text-[var(--muted)] mb-3">
          About this page
        </p>

        <h2 className="font-display text-[2rem] font-light text-[var(--text)] leading-tight mb-6">
          A life in cinema,<br />
          <em style={{ fontStyle: 'italic', color: 'var(--blue)' }}>made visible.</em>
        </h2>

        <div className="space-y-4">
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            I watch a lot of movies. Whenever I tell people that, the same two questions follow: <em>&ldquo;Can I see the list?&rdquo;</em> and <em>&ldquo;What should I watch tonight?&rdquo;</em>
          </p>
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            This page is the answer to both. It&apos;s a personal archive of every film I&apos;ve watched since 2019 — searchable, filterable, and paired with an AI that knows the collection inside out and can recommend from it.
          </p>
          <p className="font-body text-[0.88rem] text-[var(--sub)] leading-relaxed">
            The stats page came later, out of curiosity. Turns out watching 800 films leaves a pattern worth looking at.
          </p>
        </div>

        <div
          className="mt-8 pt-6 flex items-center justify-between"
          style={{ borderTop: '1px solid rgba(0,0,0,0.07)' }}
        >
          <p className="font-body text-[0.68rem] text-[var(--muted)]">Built with Next.js · TMDb data</p>
          <p className="font-body text-[0.68rem] text-[var(--muted)]">Since 2019</p>
        </div>
      </div>
    </div>
  )
}
