'use client'

import { useState } from 'react'

export default function FeedbackWidget() {
  const [open, setOpen]       = useState(false)
  const [text, setText]       = useState('')
  const [status, setStatus]   = useState<'idle'|'sending'|'sent'|'error'>('idle')

  const submit = async () => {
    if (!text.trim() || status === 'sending') return
    setStatus('sending')
    try {
      const res = await fetch('https://api.github.com/repos/JSam-M/movies-dashboard/issues', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/vnd.github+json' },
        body: JSON.stringify({
          title: `Feedback: ${text.slice(0, 60)}${text.length > 60 ? '…' : ''}`,
          body: text,
          labels: ['feedback'],
        }),
      })
      if (res.ok || res.status === 201) {
        setStatus('sent')
        setText('')
        setTimeout(() => { setOpen(false); setStatus('idle') }, 2500)
      } else {
        setStatus('error')
      }
    } catch {
      setStatus('error')
    }
  }

  return (
    <>
      {/* Trigger — subtle question mark, bottom-left */}
      <button
        onClick={() => setOpen(true)}
        title="Share feedback"
        className="fixed bottom-8 left-8 z-50 w-9 h-9 rounded-full flex items-center justify-center transition-all hover:scale-110"
        style={{
          background: 'rgba(0,0,0,0.06)',
          border: '1px solid rgba(0,0,0,0.08)',
          color: 'var(--muted)',
          fontSize: '14px',
          fontFamily: 'Georgia, serif',
          backdropFilter: 'blur(8px)',
        }}
      >
        ?
      </button>

      {/* Modal */}
      {open && (
        <div
          className="fixed inset-0 z-[200] flex items-end justify-start p-8"
          style={{ pointerEvents: 'none' }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0"
            style={{ pointerEvents: 'auto' }}
            onClick={() => { setOpen(false); setStatus('idle') }}
          />

          {/* Panel */}
          <div
            className="relative animate-fade-up"
            style={{
              pointerEvents: 'auto',
              width: '320px',
              background: 'rgba(255,255,255,0.97)',
              backdropFilter: 'blur(40px) saturate(1.8)',
              border: '1px solid rgba(255,255,255,0.9)',
              borderRadius: '20px',
              padding: '24px',
              boxShadow: '0 24px 64px rgba(0,0,0,0.14), 0 4px 16px rgba(0,0,0,0.06)',
            }}
          >
            <button
              onClick={() => { setOpen(false); setStatus('idle') }}
              className="absolute top-4 right-4 text-[var(--muted)] hover:text-[var(--text)] transition-colors"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>

            {status === 'sent' ? (
              <div className="text-center py-4">
                <div className="text-2xl mb-2">✓</div>
                <p className="font-body text-[0.82rem] text-[var(--text)] font-medium">Thanks for the feedback.</p>
                <p className="font-body text-[0.72rem] text-[var(--muted)] mt-1">It means a lot.</p>
              </div>
            ) : (
              <>
                <p className="font-display text-[1.1rem] font-light text-[var(--text)] mb-1">Share a thought</p>
                <p className="font-body text-[0.72rem] text-[var(--muted)] mb-4">
                  Logo, colours, filters, anything — I read every one.
                </p>
                <textarea
                  value={text}
                  onChange={e => setText(e.target.value)}
                  placeholder="Your feedback…"
                  rows={4}
                  className="w-full resize-none outline-none font-body text-[0.82rem] text-[var(--text)] rounded-xl p-3"
                  style={{
                    background: 'rgba(0,0,0,0.04)',
                    border: '1px solid rgba(0,0,0,0.08)',
                    fontFamily: 'inherit',
                  }}
                />
                {status === 'error' && (
                  <p className="font-body text-[0.7rem] text-red-500 mt-1">Something went wrong. Try again.</p>
                )}
                <button
                  onClick={submit}
                  disabled={!text.trim() || status === 'sending'}
                  className="mt-3 w-full py-2.5 rounded-xl font-body text-[0.82rem] font-medium transition-all"
                  style={{
                    background: text.trim() ? '#0071e3' : 'rgba(0,0,0,0.06)',
                    color: text.trim() ? 'white' : 'var(--muted)',
                    cursor: text.trim() ? 'pointer' : 'default',
                  }}
                >
                  {status === 'sending' ? 'Sending…' : 'Send'}
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}
