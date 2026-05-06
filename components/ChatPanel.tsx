'use client'

import { useState, useRef, useEffect } from 'react'
import type { Movie } from '@/lib/movies'

interface Message { role: 'user'|'assistant'; content: string }

interface Props { movies: Movie[]; onClose: () => void }

const QUICK_PROMPTS = [
  "I loved Parasite — what should I watch?",
  "Recommend Tamil films from your collection",
  "Something feel-good for tonight",
  "Best films under 2 hours",
  "Hidden gems with high ratings",
  "Great films to watch with family",
]

function formatMessage(text: string) {
  // Bold **text**
  const parts = text.split(/(\*\*[^*]+\*\*)/g)
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**'))
      return <strong key={i} className="font-semibold text-[var(--text)]">{p.slice(2,-2)}</strong>
    return <span key={i}>{p}</span>
  })
}

export default function ChatPanel({ movies, onClose }: Props) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `Hi! I can recommend films from this collection of ${movies.length} films based on your taste.\n\nTell me what you enjoy — name a few films you love, a mood you're in, or a genre you prefer. I'll find the perfect match from what's been watched here.`
    }
  ])
  const [input, setInput]     = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef             = useRef<HTMLDivElement>(null)
  const inputRef              = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    inputRef.current?.focus()
  }, [])

  const send = async (text?: string) => {
    const content = text || input.trim()
    if (!content || loading) return
    setInput('')

    const userMsg: Message = { role: 'user', content }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: newMessages.map(m => ({ role: m.role, content: m.content })),
        }),
      })
      const data = await res.json()
      setMessages(prev => [...prev, { role: 'assistant', content: data.message || 'Sorry, something went wrong.' }])
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Connection error. Please try again.' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-end p-6 pointer-events-none">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/20 backdrop-blur-sm pointer-events-auto" onClick={onClose} />

      {/* Panel */}
      <div className="relative pointer-events-auto flex flex-col rounded-3xl overflow-hidden"
        style={{
          width: '420px', height: '680px',
          background: 'rgba(255,255,255,0.92)',
          backdropFilter: 'blur(40px) saturate(1.8)',
          border: '1px solid rgba(255,255,255,0.9)',
          boxShadow: '0 32px 80px rgba(0,0,0,0.18), 0 8px 24px rgba(0,0,0,0.08)',
          animation: 'fadeUp 0.35s cubic-bezier(0.16,1,0.3,1) forwards',
        }}>

        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-black/7">
          <div className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg,#0071e3,#34aadc)' }}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
          </div>
          <div>
            <p className="font-body text-[0.85rem] font-semibold text-[var(--text)]">Film Recommender</p>
            <p className="font-body text-[0.68rem] text-[var(--sub)]">{movies.length} films in collection</p>
          </div>
          <button onClick={onClose} className="ml-auto p-1.5 rounded-full hover:bg-black/5 transition-colors">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
            </svg>
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role==='user' ? 'justify-end' : 'justify-start'}`}>
              {m.role==='assistant' && (
                <div className="w-6 h-6 rounded-full mr-2 mt-1 flex-shrink-0"
                  style={{ background: 'linear-gradient(135deg,#0071e3,#34aadc)' }} />
              )}
              <div className={`max-w-[85%] px-4 py-3 font-body text-[0.83rem] leading-relaxed ${m.role==='user' ? 'chat-bubble-user' : 'chat-bubble-ai'}`}
                style={{ whiteSpace: 'pre-wrap' }}>
                {formatMessage(m.content)}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="w-6 h-6 rounded-full mr-2 flex-shrink-0"
                style={{ background: 'linear-gradient(135deg,#0071e3,#34aadc)' }} />
              <div className="chat-bubble-ai px-4 py-3">
                <div className="flex gap-1.5 items-center h-4">
                  {[0,1,2].map(i => (
                    <div key={i} className="w-1.5 h-1.5 rounded-full bg-blue-400"
                      style={{ animation: `pulse-dot 1.2s ease ${i*0.2}s infinite` }} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Quick prompts — show only at start */}
          {messages.length === 1 && (
            <div className="pt-2 space-y-2">
              <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--muted)] px-1">
                Quick start
              </p>
              <div className="flex flex-wrap gap-2">
                {QUICK_PROMPTS.map(q => (
                  <button key={q} onClick={() => send(q)}
                    className="px-3 py-1.5 rounded-full font-body text-[0.75rem] text-[var(--sub)] transition-all hover:bg-white hover:shadow-sm"
                    style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div className="px-4 pb-4 pt-2 border-t border-black/7">
          <div className="flex gap-2 items-center">
            <input
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && send()}
              placeholder="Name a film you love…"
              className="flex-1 px-4 py-2.5 rounded-2xl font-body text-[0.85rem] text-[var(--text)] outline-none transition-all"
              style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}
            />
            <button
              onClick={() => send()}
              disabled={!input.trim() || loading}
              className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-all"
              style={{
                background: input.trim() && !loading ? 'linear-gradient(135deg,#0071e3,#34aadc)' : 'rgba(0,0,0,0.08)',
                boxShadow: input.trim() && !loading ? '0 2px 8px rgba(0,113,227,0.3)' : 'none',
              }}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={input.trim()&&!loading?'white':'#86868b'} strokeWidth="2.5">
                <line x1="22" y1="2" x2="11" y2="13"/>
                <polygon points="22 2 15 22 11 13 2 9 22 2"/>
              </svg>
            </button>
          </div>
          <p className="font-body text-[0.62rem] text-[var(--muted)] text-center mt-2">
            Recommends only from films in this collection
          </p>
        </div>
      </div>
    </div>
  )
}
