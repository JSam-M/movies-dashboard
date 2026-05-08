'use client'

import { useState, useEffect } from 'react'
import type { Movie } from '@/lib/movies'
import ChatPanel from '@/components/ChatPanel'
import Link from 'next/link'

export default function DiscoverPage() {
  const [allMovies, setAllMovies] = useState<Movie[]>([])
  const [filtered,  setFiltered]  = useState<Movie[]>([])
  const [loading,   setLoading]   = useState(true)
  const [chatOpen,  setChatOpen]  = useState(false)
  const [search,    setSearch]    = useState('')
  const [genre,     setGenre]     = useState('All')
  const [language,  setLanguage]  = useState('All')
  const [showRewatched, setShowRewatched] = useState(false)
  const [sortBy,    setSortBy]    = useState<'rating'|'rewatched'|'recent'>('rating')
  const [stats,     setStats]     = useState<Record<string,unknown>>({})

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, stats: s }) => {
        setAllMovies(movies); setStats(s); setLoading(false)
      })
  }, [])

  useEffect(() => {
    let f = [...allMovies]
    if (search)        f = f.filter(m => m.name.toLowerCase().includes(search.toLowerCase()) || m.director.toLowerCase().includes(search.toLowerCase()))
    if (genre !== 'All')    f = f.filter(m => m.genre.includes(genre))
    if (language !== 'All') f = f.filter(m => m.language === language)
    if (showRewatched)      f = f.filter(m => m.timesWatched >= 2)
    if (sortBy === 'rating')    f = [...f].sort((a,b) => b.tmdbRating - a.tmdbRating)
    if (sortBy === 'rewatched') f = [...f].sort((a,b) => b.timesWatched - a.timesWatched)
    if (sortBy === 'recent')    f = [...f].sort((a,b) => b.date.localeCompare(a.date))
    setFiltered(f)
  }, [search, genre, language, showRewatched, sortBy, allMovies])

  const genres    = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()]
  const languages = ['All', ...Array.from(new Set(allMovies.map(m => m.language))).sort()]

  // Top picks: mix of highest rated + most rewatched, deduplicated
  const topRated    = [...allMovies].sort((a,b) => b.tmdbRating - a.tmdbRating).slice(0,4)
  const mostRewatched = [...allMovies].filter(m => m.timesWatched >= 2).sort((a,b) => b.timesWatched - a.timesWatched).slice(0,2)
  const topPicks    = Array.from(new Map([...topRated, ...mostRewatched].map(m => [m.name, m])).values()).slice(0,6)

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2">
        {[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full bg-blue-300" style={{animation:`pulse-dot 1.2s ease ${i*0.2}s infinite`}} />)}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen mesh-bg">
      {/* NAV */}
      <nav className="sticky top-0 z-40 border-b border-black/7" style={{background:'rgba(245,245,247,0.85)',backdropFilter:'blur(20px)'}}>
        <div className="max-w-[1200px] mx-auto px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-lg">🎬</span>
            <span className="font-display text-lg font-light text-[var(--text)]">Film Collection</span>
          </div>
          <Link href="/stats" className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors flex items-center gap-1.5">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            My Stats
          </Link>
        </div>
      </nav>

      <div className="max-w-[1200px] mx-auto px-8 py-20">
        {/* HERO */}
        <div className="text-center mb-20">
          <p className="font-body text-[0.65rem] font-semibold tracking-[0.2em] uppercase text-[var(--sub)] mb-5">Personal Film Archive · Since 2019</p>
          <h1 className="font-display text-[clamp(3.5rem,7vw,6.5rem)] font-light leading-[0.9] tracking-tight text-[var(--text)] mb-8">
            A life in{' '}
            <em style={{fontStyle:'italic',background:'linear-gradient(135deg,#0071e3,#34aadc)',WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>cinema</em>
          </h1>
          <p className="font-body text-[1rem] text-[var(--sub)] max-w-md mx-auto leading-relaxed mb-10">
            {stats.total as number} films watched. Not sure what to watch? The AI knows this collection inside out.
          </p>
        </div>

        {/* TOP PICKS */}
        <div className="mb-16">
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Curated</p>
              <p className="font-display text-[2rem] font-light text-[var(--text)]">Top Picks</p>
            </div>
            <p className="font-body text-[0.72rem] text-[var(--muted)]">Highest rated + most rewatched</p>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {topPicks.map(m => (
              <div key={m.name} className="glass rounded-2xl p-5 hover:shadow-lg transition-all group">
                <div className="flex items-start justify-between mb-3">
                  <span className="font-body text-[0.62rem] font-semibold tracking-[0.08em] uppercase px-2 py-1 rounded-full"
                    style={{background:'rgba(0,113,227,0.07)',color:'var(--blue)'}}>
                    {m.genre.split(',')[0].trim()}
                  </span>
                  <div className="text-right">
                    <span className="font-display text-[1.3rem] font-light" style={{color:'var(--blue)'}}>{m.tmdbRating.toFixed(1)}</span>
                    {m.timesWatched >= 2 && <span className="block font-body text-[0.6rem] text-amber-500 font-semibold">{m.timesWatched}× watched</span>}
                  </div>
                </div>
                <p className="font-display text-[1.1rem] font-light text-[var(--text)] leading-tight mb-1">{m.name}</p>
                <p className="font-body text-[0.7rem] text-[var(--sub)] mb-3">{m.releaseYear} · {m.language} · {m.runtime}</p>
                <p className="font-body text-[0.76rem] text-[var(--sub)] leading-relaxed line-clamp-3">{m.overview}</p>
              </div>
            ))}
          </div>
        </div>

        {/* BROWSE */}
        <div>
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Browse</p>
              <p className="font-display text-[2rem] font-light text-[var(--text)]">Full Collection</p>
            </div>
            <p className="font-body text-[0.75rem] text-[var(--muted)]">{filtered.length} films</p>
          </div>

          {/* Filters row */}
          <div className="flex gap-3 mb-4 flex-wrap items-center">
            <div className="relative flex-1 min-w-[180px]">
              <svg className="absolute left-3 top-1/2 -translate-y-1/2" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#86868b" strokeWidth="2">
                <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search films or directors…"
                className="w-full pl-9 pr-4 py-2.5 rounded-xl font-body text-sm outline-none"
                style={{background:'white',border:'1px solid rgba(0,0,0,0.08)',color:'var(--text)'}} />
            </div>
            <select value={genre} onChange={e => setGenre(e.target.value)}
              className="px-3 py-2.5 rounded-xl font-body text-sm outline-none"
              style={{background:'white',border:'1px solid rgba(0,0,0,0.08)',color:'var(--text)'}}>
              {genres.slice(0,25).map(g => <option key={g}>{g}</option>)}
            </select>
            <select value={language} onChange={e => setLanguage(e.target.value)}
              className="px-3 py-2.5 rounded-xl font-body text-sm outline-none"
              style={{background:'white',border:'1px solid rgba(0,0,0,0.08)',color:'var(--text)'}}>
              {languages.map(l => <option key={l}>{l}</option>)}
            </select>
            <select value={sortBy} onChange={e => setSortBy(e.target.value as 'rating'|'rewatched'|'recent')}
              className="px-3 py-2.5 rounded-xl font-body text-sm outline-none"
              style={{background:'white',border:'1px solid rgba(0,0,0,0.08)',color:'var(--text)'}}>
              <option value="rating">Sort: Rating</option>
              <option value="rewatched">Sort: Rewatched</option>
              <option value="recent">Sort: Recent</option>
            </select>
            <button onClick={() => setShowRewatched(!showRewatched)}
              className="px-4 py-2.5 rounded-xl font-body text-sm font-medium transition-all"
              style={{
                background: showRewatched ? 'var(--blue)' : 'white',
                color: showRewatched ? 'white' : 'var(--sub)',
                border: `1px solid ${showRewatched ? 'var(--blue)' : 'rgba(0,0,0,0.08)'}`,
              }}>
              ★ Favourites
            </button>
          </div>

          {/* Film list */}
          <div className="grid grid-cols-1 gap-2">
            {filtered.slice(0,60).map(m => (
              <div key={m.name} className="glass rounded-xl px-5 py-3.5 flex items-center gap-5 hover:bg-white/90 transition-all">
                <div className="font-display text-[1.5rem] font-light w-12 text-center flex-shrink-0" style={{color:'var(--blue)'}}>
                  {m.tmdbRating > 0 ? m.tmdbRating.toFixed(1) : '—'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-body text-[0.88rem] font-medium text-[var(--text)] truncate">
                    {m.name} {m.timesWatched >= 2 && <span className="text-amber-400 ml-1" title={`Watched ${m.timesWatched}×`}>★</span>}
                  </p>
                  <p className="font-body text-[0.7rem] text-[var(--sub)]">{m.releaseYear} · {m.director.split(',')[0].trim()} · {m.runtime}</p>
                </div>
                <div className="hidden md:flex gap-2 flex-shrink-0 items-center">
                  {m.timesWatched >= 2 && (
                    <span className="font-body text-[0.62rem] px-2 py-1 rounded-full font-semibold" style={{background:'rgba(251,191,36,0.12)',color:'#d97706'}}>{m.timesWatched}×</span>
                  )}
                  <span className="font-body text-[0.62rem] px-2 py-1 rounded-full" style={{background:'rgba(0,0,0,0.04)',color:'var(--sub)'}}>{m.language}</span>
                  <span className="font-body text-[0.62rem] px-2 py-1 rounded-full" style={{background:'rgba(0,0,0,0.04)',color:'var(--sub)'}}>{m.genre.split(',')[0].trim()}</span>
                </div>
              </div>
            ))}
            {filtered.length > 60 && (
              <p className="text-center font-body text-[0.75rem] text-[var(--muted)] py-4">Showing 60 of {filtered.length} — refine filters to narrow down</p>
            )}
          </div>
        </div>

        <div className="mt-16 pt-6 border-t border-black/7 text-center">
          <p className="font-body text-[0.65rem] tracking-[0.1em] uppercase text-[rgba(0,0,0,0.2)]">
            {stats.total as number} films · v2.0 · {new Date().toLocaleDateString('en-US',{month:'long',year:'numeric'})}
          </p>
        </div>
      </div>

      {/* FLOATING CHAT */}
      <button onClick={() => setChatOpen(true)}
        className="fixed bottom-8 right-8 z-50 w-14 h-14 rounded-full flex items-center justify-center text-white shadow-2xl transition-all hover:scale-105"
        style={{background:'linear-gradient(135deg,#0071e3,#34aadc)',boxShadow:'0 8px 32px rgba(0,113,227,0.4)'}}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>

      {chatOpen && <ChatPanel movies={allMovies} onClose={() => setChatOpen(false)} />}
    </div>
  )
}
