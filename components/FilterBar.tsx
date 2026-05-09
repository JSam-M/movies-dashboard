'use client'

import { useState } from 'react'

interface Props {
  search: string; setSearch: (v: string) => void
  language: string; setLanguage: (v: string) => void; languages: string[]
  genre: string; setGenre: (v: string) => void; genres: string[]
  director: string; setDirector: (v: string) => void; directors: string[]
  minRating: number; setMinRating: (v: number) => void
  minWY: number; maxWY: number
  watchYear: [number,number] | null; setWatchYear: (v: [number,number] | null) => void
  rewatchFilter: string; setRewatchFilter: (v: string) => void
  total: number; filtered: number
}

export default function FilterBar(p: Props) {
  const [open, setOpen] = useState(false)

  const activeCount = [
    p.search, p.language !== 'All', p.genre !== 'All', p.director !== 'All',
    p.minRating > 0, p.watchYear !== null, p.rewatchFilter !== 'All'
  ].filter(Boolean).length

  const reset = () => {
    p.setSearch(''); p.setLanguage('All'); p.setGenre('All')
    p.setDirector('All'); p.setMinRating(0); p.setWatchYear(null); p.setRewatchFilter('All')
  }

  return (
    <div className="mb-8">
      <div className="flex items-center gap-3 mb-3">
        <button onClick={() => setOpen(!open)}
          className="flex items-center gap-2 px-4 py-2 rounded-full font-body text-[0.75rem] font-medium transition-all"
          style={{
            background: open ? 'var(--blue)' : 'white',
            color: open ? 'white' : 'var(--sub)',
            border: `1px solid ${open ? 'var(--blue)' : 'rgba(0,0,0,0.1)'}`,
            boxShadow: '0 1px 4px rgba(0,0,0,0.06)',
          }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
          </svg>
          Filters
          {activeCount > 0 && (
            <span className="ml-1 px-1.5 py-0.5 rounded-full text-[0.6rem] font-semibold"
              style={{ background: open ? 'rgba(255,255,255,0.25)' : 'var(--blue)', color: open ? 'white' : 'white' }}>
              {activeCount}
            </span>
          )}
        </button>

        {activeCount > 0 && (
          <button onClick={reset} className="font-body text-[0.7rem] text-[var(--sub)] hover:text-[var(--text)] transition-colors">
            Clear all
          </button>
        )}

        <span className="ml-auto font-body text-[0.7rem] text-[var(--muted)]">
          {p.filtered} of {p.total} films
        </span>
      </div>

      {open && (
        <div className="glass rounded-2xl p-6 animate-fade-in">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5">

            {/* Search */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">Search</label>
              <input value={p.search} onChange={e => p.setSearch(e.target.value)}
                placeholder="Film title…"
                className="w-full px-3 py-2 rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none"
                style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }} />
            </div>

            {/* Language */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">Language</label>
              <select value={p.language} onChange={e => p.setLanguage(e.target.value)}
                className="w-full px-3 py-2 rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none"
                style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}>
                {p.languages.map(l => <option key={l}>{l}</option>)}
              </select>
            </div>

            {/* Genre */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">Genre</label>
              <select value={p.genre} onChange={e => p.setGenre(e.target.value)}
                className="w-full px-3 py-2 rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none"
                style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}>
                {p.genres.map(g => <option key={g}>{g}</option>)}
              </select>
            </div>

            {/* Min Rating */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">
                Min Rating — {p.minRating.toFixed(1)}
              </label>
              <input type="range" min="0" max="10" step="0.5" value={p.minRating}
                onChange={e => p.setMinRating(parseFloat(e.target.value))}
                className="w-full" style={{accentColor:'var(--blue)'}} />
            </div>

            {/* Watch Year */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">
                Watch Year — {p.watchYear ? `${p.watchYear[0]}–${p.watchYear[1]}` : 'All'}
              </label>
              <div className="flex gap-2">
                <select
                  value={p.watchYear?.[0] ?? p.minWY}
                  onChange={e => p.setWatchYear([parseInt(e.target.value), p.watchYear?.[1] ?? p.maxWY])}
                  className="flex-1 px-2 py-2 rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none"
                  style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}>
                  {Array.from({ length: p.maxWY - p.minWY + 1 }, (_, i) => p.minWY + i).map(y => (
                    <option key={y}>{y}</option>
                  ))}
                </select>
                <select
                  value={p.watchYear?.[1] ?? p.maxWY}
                  onChange={e => p.setWatchYear([p.watchYear?.[0] ?? p.minWY, parseInt(e.target.value)])}
                  className="flex-1 px-2 py-2 rounded-xl font-body text-[0.85rem] text-[var(--text)] outline-none"
                  style={{ background: 'rgba(0,0,0,0.04)', border: '1px solid rgba(0,0,0,0.08)' }}>
                  {Array.from({ length: p.maxWY - p.minWY + 1 }, (_, i) => p.minWY + i).map(y => (
                    <option key={y}>{y}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* View */}
            <div>
              <label className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] block mb-2">View</label>
              <div className="flex gap-2">
                {['All','Rewatched','First watch'].map(opt => (
                  <button key={opt} onClick={() => p.setRewatchFilter(opt)}
                    className="flex-1 py-1.5 rounded-full font-body text-[0.7rem] font-medium transition-all"
                    style={{
                      background: p.rewatchFilter === opt ? 'var(--blue)' : 'rgba(0,0,0,0.04)',
                      color: p.rewatchFilter === opt ? 'white' : 'var(--sub)',
                      border: `1px solid ${p.rewatchFilter === opt ? 'var(--blue)' : 'rgba(0,0,0,0.08)'}`,
                    }}>
                    {opt}
                  </button>
                ))}
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  )
}
