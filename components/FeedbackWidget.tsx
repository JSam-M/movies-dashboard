'use client'

import { useState } from 'react'

export default function FeedbackWidget() {
  const [open, setOpen]       = useState(false)
  const [text, setText]       = useState('')
  const [status, setStatus]   = useState<'idle'|'sending'|'done'|'error'>('idle')

  const submit = async () => {
    if (!text.trim() || status === 'sending') return
    setStatus('sending')
    try {
      const res = await fetch('https://api.github.com/repos/JSam-M/movies-dashboard/issues', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/vnd.github+json' },
        body: JSON.stringify({
          title: 'Feedback',
          body: text.trim(),
          labels: ['feedback'],
        }),
      })
      if (res.ok) {
        setStatus('done')
        setText('')
        setTimeout(() => { setOpen(false); setStatus('idle') }, 2500)
      } else {
        setStatus('error')
        setTimeout(() => setStatus('idle'), 3000)
      }
    } catch {
      setStatus('error')
      setTimeout(() => setStatus('idle'), 3000)
    }
  }

  return (
    <>
      {/* Trigger — subtle ? icon, bottom-left */}
      <button
        onClick={() => setOpen(true)}
        title="Send feedback"
        className="fixed bottom-8 left-8 z-50 w-9 h-9 rounded-full flex items-center justify-center transition-all hover:scale-[1.04]"
        style={{
          background: 'rgba(255,255,255,0.7)',
          border: '1px solid rgba(0,0,0,0.08)',
          backdropFilter: 'blur(12px)',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          color: 'var(--muted)',
          fontSize: '0.85rem',
          fontFamily: 'Georgia, serif',
          fontStyle: 'italic',
        }}
      >
        ?
      </button>

      {/* Modal overlay */}
      {open && (
        <div
          className="fixed inset-0 z-[200] flex items-end sm:items-center justify-center p-6"
          style={{ background: 'rgba(0,0,0,0.15)', backdropFilter: 'blur(8px)' }}
          onClick={() => { setOpen(false); setStatus('idle') }}
        >
          <div
            className="w-full animate-fade-up rounded-3xl"
            style={{
              maxWidth: '400px',
              background: 'rgba(255,255,255,0.97)',
              padding: '28px',
              boxShadow: '0 32px 80px rgba(0,0,0,0.18), 0 8px 24px rgba(0,0,0,0.08)',
              border: '1px solid rgba(255,255,255,0.9)',
            }}
            onClick={e => e.stopPropagation()}
          >
            {status === 'done' ? (
              <div className="text-center py-4">
                <div className="text-2xl mb-3">✓</div>
                <p className="font-display text-[1rem] font-light text-[var(--text)]">Thank you</p>
                <p className="font-body text-[0.75rem] text-[var(--sub)] mt-1">Your feedback was received.</p>
              </div>
            ) : (
              <>
                <div className="mb-5">
                  <p className="font-display text-[1.5rem] font-light text-[var(--text)]">Share feedback</p>
                  <p className="font-body text-[0.7rem] text-[var(--sub)] mt-1">Anything — the design, a film missing, a bug, whatever.</p>
                </div>
                <textarea
                  value={text}
                  onChange={e => setText(e.target.value)}
                  placeholder="Your thoughts…"
                  rows={4}
                  className="w-full rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none resize-none"
                  style={{
                    background: 'rgba(0,0,0,0.04)',
                    border: '1px solid rgba(0,0,0,0.08)',
                    padding: '12px 14px',
                    fontFamily: 'inherit',
                  }}
                />
                {status === 'error' && (
                  <p className="font-body text-[0.7rem] mt-2" style={{color:'#ff3b30'}}>Something went wrong. Try again.</p>
                )}
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={() => { setOpen(false); setStatus('idle') }}
                    className="flex-1 py-2.5 rounded-xl font-body text-[0.75rem] font-medium transition-all"
                    style={{ background: 'rgba(0,0,0,0.05)', color: 'var(--sub)' }}
                  >
                    Cancel
                  </button>
                  <button
                    onClick={submit}
                    disabled={!text.trim() || status === 'sending'}
                    className="flex-1 py-2.5 rounded-xl font-body text-[0.75rem] font-medium text-white transition-all"
                    style={{
                      background: text.trim() && status !== 'sending' ? '#0071e3' : 'rgba(0,0,0,0.12)',
                      color: text.trim() && status !== 'sending' ? 'white' : 'var(--muted)',
                    }}
                  >
                    {status === 'sending' ? 'Sending…' : 'Send'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  )
}
