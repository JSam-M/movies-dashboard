'use client'

import { useState, useEffect, useCallback } from 'react'
import type { Movie } from '@/lib/movies'
import KPIBar from '@/components/KPIBar'
import CatalogueTab from '@/components/CatalogueTab'
import RankingsTab from '@/components/RankingsTab'
import CompositionTab from '@/components/CompositionTab'
import TrendsTab from '@/components/TrendsTab'
import ChatPanel from '@/components/ChatPanel'
import Link from 'next/link'

export default function StatsPage() {
  const [allMovies,  setAllMovies]  = useState<Movie[]>([])
  const [allEntries, setAllEntries] = useState<Movie[]>([])
  const [filtered,   setFiltered]   = useState<Movie[]>([])
  const [stats,      setStats]      = useState<Record<string,unknown>>({})
  const [activeTab,  setActiveTab]  = useState('catalogue')
  const [chatOpen,   setChatOpen]   = useState(false)
  const [loading,    setLoading]    = useState(true)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Filters
  const [search,       setSearch]       = useState('')
  const [language,     setLanguage]     = useState('All')
  const [genre,        setGenre]        = useState('All')
  const [director,     setDirector]     = useState('All')
  const [minRating,    setMinRating]    = useState(0)
  const [watchYear,    setWatchYear]    = useState<[number,number] | null>(null)
  const [rewatchFilter,setRewatchFilter]= useState('All')

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, allEntries: ae, stats: s }) => {
        setAllMovies(movies); setAllEntries(ae); setStats(s)
        setFiltered(movies); setLoading(false)
      })
  }, [])

  const applyFilters = useCallback(() => {
    let f = [...allMovies]
    if (search)             f = f.filter(m => m.name.toLowerCase().includes(search.toLowerCase()))
    if (language !== 'All') f = f.filter(m => m.language === language)
    if (genre !== 'All')    f = f.filter(m => m.genre.includes(genre))
    if (director !== 'All') f = f.filter(m => m.director.includes(director))
    if (minRating > 0)      f = f.filter(m => m.tmdbRating >= minRating)
    if (watchYear) {
      const names = new Set(allEntries.filter(e => {
        const y = parseInt('20' + e.date.split('/')[2])
        return y >= watchYear[0] && y <= watchYear[1]
      }).map(e => e.name))
      f = f.filter(m => names.has(m.name))
    }
    if (rewatchFilter === 'Rewatched')   f = f.filter(m => m.timesWatched >= 2)
    if (rewatchFilter === 'First watch') f = f.filter(m => m.timesWatched <= 1)
    setFiltered(f)
  }, [allMovies, allEntries, search, language, genre, director, minRating, watchYear, rewatchFilter])

  useEffect(() => { applyFilters() }, [applyFilters])

  const languages = ['All', ...Array.from(new Set(allMovies.map(m => m.language))).sort()]
  const genres    = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()]
  const directors = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.director.split(',').map(d => d.trim()).filter(d => d && d !== 'N/A')))).sort()]
  const watchYears = Array.from(new Set(allEntries.map(e => parseInt('20' + e.date.split('/')[2])).filter(y => !isNaN(y)))).sort()
  const minWY = watchYears[0] || 2019
  const maxWY = watchYears[watchYears.length-1] || 2026

  const activeFilters = [search, language!=='All', genre!=='All', director!=='All', minRating>0, watchYear!==null, rewatchFilter!=='All'].filter(Boolean).length

  const resetFilters = () => {
    setSearch(''); setLanguage('All'); setGenre('All')
    setDirector('All'); setMinRating(0); setWatchYear(null); setRewatchFilter('All')
  }

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2 justify-center">
        {[0,1,2].map(i => (
          <div key={i} className="w-2 h-2 rounded-full bg-blue-300"
            style={{ animation: `pulse-dot 1.2s ease ${i*0.2}s infinite` }} />
        ))}
      </div>
    </div>
  )

  const inputStyle = { background:'rgba(0,0,0,0.04)', border:'1px solid rgba(0,0,0,0.08)', borderRadius:'10px', padding:'8px 10px', fontFamily:'inherit', fontSize:'0.78rem', color:'var(--text)', width:'100%', outline:'none' }
  const labelStyle = { display:'block', fontFamily:'inherit', fontSize:'0.6rem', fontWeight:600, letterSpacing:'0.12em', textTransform:'uppercase' as const, color:'var(--sub)', marginBottom:'6px' }

  return (
    <div className="min-h-screen mesh-bg flex flex-col">
      {/* NAV */}
      <nav className="sticky top-0 z-40 border-b border-black/7 flex-shrink-0"
        style={{ background:'rgba(245,245,247,0.85)', backdropFilter:'blur(20px)' }}>
        <div className="max-w-[1600px] mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2 hover:opacity-70 transition-opacity">
              <span className="text-lg">🎬</span>
              <span className="font-display text-lg font-light text-[var(--text)]">Film Collection</span>
            </Link>
            <span className="text-black/20 text-sm">/</span>
            <span className="font-body text-[0.75rem] font-semibold text-[var(--sub)]">Stats</span>
          </div>
          <div className="flex items-center gap-3">
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg font-body text-[0.72rem] font-medium transition-all"
              style={{ background: sidebarOpen ? 'rgba(0,113,227,0.08)' : 'rgba(0,0,0,0.04)', color: sidebarOpen ? 'var(--blue)' : 'var(--sub)', border: `1px solid ${sidebarOpen ? 'rgba(0,113,227,0.2)' : 'rgba(0,0,0,0.08)'}` }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
              </svg>
              Filters {activeFilters > 0 && <span className="ml-1 w-4 h-4 rounded-full text-[0.6rem] font-bold bg-blue-500 text-white flex items-center justify-center">{activeFilters}</span>}
            </button>
            <button onClick={() => setChatOpen(true)}
              className="flex items-center gap-2 px-4 py-2 rounded-full font-body text-[0.75rem] font-semibold text-white"
              style={{ background:'linear-gradient(135deg,#0071e3,#34aadc)', boxShadow:'0 2px 10px rgba(0,113,227,0.3)' }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              Ask AI
            </button>
          </div>
        </div>
      </nav>

      <div className="flex flex-1 max-w-[1600px] mx-auto w-full">
        {/* SIDEBAR */}
        {sidebarOpen && (
          <aside className="w-64 flex-shrink-0 border-r border-black/7 p-5 overflow-y-auto"
            style={{ background:'rgba(255,255,255,0.6)', backdropFilter:'blur(20px)' }}>
            <div className="flex items-center justify-between mb-5">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.14em] uppercase text-[var(--sub)]">Refine</p>
              {activeFilters > 0 && (
                <button onClick={resetFilters} className="font-body text-[0.65rem] text-[var(--blue)] hover:opacity-70">Clear all</button>
              )}
            </div>

            <div className="space-y-5">
              <div>
                <label style={labelStyle}>Search</label>
                <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Film title…" style={inputStyle} />
              </div>

              <div>
                <label style={labelStyle}>Language</label>
                <select value={language} onChange={e => setLanguage(e.target.value)} style={inputStyle}>
                  {languages.map(l => <option key={l}>{l}</option>)}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Genre</label>
                <select value={genre} onChange={e => setGenre(e.target.value)} style={inputStyle}>
                  {genres.map(g => <option key={g}>{g}</option>)}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Director</label>
                <select value={director} onChange={e => setDirector(e.target.value)} style={inputStyle}>
                  {directors.slice(0,100).map(d => <option key={d}>{d}</option>)}
                </select>
              </div>

              <div>
                <label style={labelStyle}>Min Rating — {minRating.toFixed(1)}</label>
                <input type="range" min="0" max="10" step="0.5" value={minRating}
                  onChange={e => setMinRating(parseFloat(e.target.value))}
                  className="w-full accent-blue-500" />
              </div>

              <div>
                <label style={labelStyle}>Watch Year</label>
                <div className="flex gap-2">
                  <select value={watchYear?.[0] ?? minWY} onChange={e => setWatchYear([parseInt(e.target.value), watchYear?.[1] ?? maxWY])}
                    style={{ ...inputStyle, width:'50%' }}>
                    {Array.from({length: maxWY-minWY+1},(_,i)=>minWY+i).map(y=><option key={y}>{y}</option>)}
                  </select>
                  <select value={watchYear?.[1] ?? maxWY} onChange={e => setWatchYear([watchYear?.[0] ?? minWY, parseInt(e.target.value)])}
                    style={{ ...inputStyle, width:'50%' }}>
                    {Array.from({length: maxWY-minWY+1},(_,i)=>minWY+i).map(y=><option key={y}>{y}</option>)}
                  </select>
                </div>
                {watchYear && (
                  <button onClick={() => setWatchYear(null)} className="font-body text-[0.65rem] text-[var(--blue)] mt-1">Clear year filter</button>
                )}
              </div>

              <div>
                <label style={labelStyle}>View</label>
                <div className="flex flex-col gap-1.5">
                  {['All','Rewatched','First watch'].map(opt => (
                    <button key={opt} onClick={() => setRewatchFilter(opt)}
                      className="py-1.5 rounded-lg font-body text-[0.72rem] font-medium text-left px-3 transition-all"
                      style={{ background: rewatchFilter===opt ? 'var(--blue)' : 'rgba(0,0,0,0.04)', color: rewatchFilter===opt ? 'white' : 'var(--sub)', border: `1px solid ${rewatchFilter===opt ? 'var(--blue)' : 'rgba(0,0,0,0.08)'}` }}>
                      {opt}
                    </button>
                  ))}
                </div>
              </div>

              <div className="pt-2 border-t border-black/7">
                <p className="font-body text-[0.7rem] text-[var(--muted)]">
                  <span style={{ color:'var(--text)', fontWeight:600 }}>{filtered.length}</span> of {allMovies.length} films
                </p>
              </div>
            </div>
          </aside>
        )}

        {/* MAIN CONTENT */}
        <main className="flex-1 min-w-0 px-8 py-8 overflow-x-hidden">
          {/* Header */}
          <div className="mb-8 pb-6 border-b border-black/7 flex items-end justify-between">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Personal Archive</p>
              <h1 className="font-display text-[2.5rem] font-light leading-tight text-[var(--text)]">
                My Film Stats
              </h1>
            </div>
            {filtered.length > 0 && (() => {
              const top = [...filtered].sort((a,b)=>b.tmdbRating-a.tmdbRating)[0]
              return (
                <div className="glass rounded-xl px-5 py-4 text-right hidden lg:block">
                  <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Highest Rated</p>
                  <p className="font-display text-[1rem] font-light text-[var(--text)]">{top.name}</p>
                  <p className="font-display text-[1.8rem] font-light" style={{ color:'var(--blue)' }}>{top.tmdbRating.toFixed(1)}<span className="font-body text-sm text-[var(--muted)]"> /10</span></p>
                </div>
              )
            })()}
          </div>

          {/* KPIs */}
          <KPIBar movies={filtered} allEntries={allEntries} watchYear={watchYear} />

          {/* Tabs */}
          <div className="mt-8 border-b border-black/7 flex gap-0">
            {['catalogue','rankings','composition','trends'].map(tab => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase px-6 py-4 border-b-[1.5px] transition-colors"
                style={{ color: activeTab===tab ? 'var(--text)' : 'rgba(0,0,0,0.3)', borderBottomColor: activeTab===tab ? 'var(--text)' : 'transparent', marginBottom:'-1px' }}>
                {tab}
              </button>
            ))}
          </div>

          <div className="mt-8">
            {activeTab === 'catalogue'   && <CatalogueTab   movies={filtered} />}
            {activeTab === 'rankings'    && <RankingsTab    movies={filtered} />}
            {activeTab === 'composition' && <CompositionTab movies={filtered} />}
            {activeTab === 'trends'      && <TrendsTab      movies={filtered} allEntries={allEntries} watchYear={watchYear} />}
          </div>

          <div className="mt-16 pt-6 border-t border-black/7 text-center">
            <p className="font-body text-[0.65rem] tracking-[0.1em] uppercase text-[rgba(0,0,0,0.2)]">
              {allMovies.length} films · v2.0 · {new Date().toLocaleDateString('en-US',{month:'long',year:'numeric'})}
            </p>
          </div>
        </main>
      </div>

      {chatOpen && <ChatPanel movies={allMovies} onClose={() => setChatOpen(false)} />}
    </div>
  )
}
