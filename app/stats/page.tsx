'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { Movie } from '@/lib/movies'
import ChatPanel from '@/components/ChatPanel'
import Link from 'next/link'
import StatsContent from '@/components/StatsContent'
import MultiSelect from '@/components/MultiSelect'
import ScrollJump from '@/components/ScrollJump'
import AboutModal from '@/components/AboutModal'
import ThemeToggle from '@/components/ThemeToggle'

export default function StatsPage() {
  const [allMovies,   setAllMovies]   = useState<Movie[]>([])
  const [allEntries,  setAllEntries]  = useState<Movie[]>([])
  const [filtered,    setFiltered]    = useState<Movie[]>([])
  const [loading,     setLoading]     = useState(true)
  const [chatOpen,    setChatOpen]    = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [aboutOpen,   setAboutOpen]   = useState(false)

  // Film multi-select
  const [filmQuery,    setFilmQuery]    = useState('')
  const [selectedFilms,setSelectedFilms]= useState<string[]>([])
  const [filmDropOpen, setFilmDropOpen] = useState(false)
  const filmRef = useRef<HTMLDivElement>(null)

  const [selLanguages,  setSelLanguages]  = useState<string[]>([])
  const [selGenres,     setSelGenres]     = useState<string[]>([])
  const [selDirectors,  setSelDirectors]  = useState<string[]>([])
  const [minRating,     setMinRating]     = useState(0)
  const [watchYears,    setWatchYears]    = useState<number[]>([])
  const [rewatchFilter, setRewatchFilter] = useState('All')

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, allEntries: ae }) => {
        setAllMovies(movies); setAllEntries(ae)
        setFiltered(movies); setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (typeof window === 'undefined' || loading) return
    const hash = window.location.hash
    if (!hash) return
    const timer = setTimeout(() => {
      const el = document.querySelector(hash)
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 800)
    return () => clearTimeout(timer)
  }, [loading])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (filmRef.current && !filmRef.current.contains(e.target as Node)) setFilmDropOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const applyFilters = useCallback(() => {
    let f = [...allMovies]
    if (selectedFilms.length)  f = f.filter(m => selectedFilms.includes(m.name))
    if (selLanguages.length)   f = f.filter(m => selLanguages.includes(m.language))
    if (selGenres.length)      f = f.filter(m => selGenres.some(g => m.genre.includes(g)))
    if (selDirectors.length)   f = f.filter(m => selDirectors.some(d => m.director.includes(d)))
    if (minRating > 0)         f = f.filter(m => m.tmdbRating >= minRating)
    if (watchYears.length > 0) {
      const names = new Set(allEntries.filter(e => {
        const y = parseInt('20' + e.date.split('/')[2])
        return watchYears.includes(y)
      }).map(e => e.name))
      f = f.filter(m => names.has(m.name))
    }
    if (rewatchFilter === 'Rewatched')   f = f.filter(m => m.timesWatched >= 2)
    if (rewatchFilter === 'First watch') f = f.filter(m => m.timesWatched <= 1)
    setFiltered(f)
  }, [allMovies, allEntries, selectedFilms, selLanguages, selGenres, selDirectors, minRating, watchYears, rewatchFilter])

  useEffect(() => { applyFilters() }, [applyFilters])

  const languages = Array.from(new Set(allMovies.map(m => m.language))).sort()
  const genres    = Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()
  const directors = Array.from(new Set(allMovies.flatMap(m => m.director.split(',').map(d => d.trim()).filter(d => d && d !== 'N/A')))).sort()
  const allYears  = Array.from(new Set(allEntries.map(e => parseInt('20' + e.date.split('/')[2])).filter(y => !isNaN(y)))).sort()

  const filmSuggestions = filmQuery.trim().length >= 1
    ? allMovies.filter(m =>
        m.name.toLowerCase().includes(filmQuery.toLowerCase()) &&
        !selectedFilms.includes(m.name)
      ).slice(0, 10)
    : []

  const activeFilters = [selectedFilms.length>0, selLanguages.length>0, selGenres.length>0, selDirectors.length>0, minRating>0, watchYears.length>0, rewatchFilter!=='All'].filter(Boolean).length

  const resetFilters = () => {
    setSelectedFilms([]); setFilmQuery('')
    setSelLanguages([]); setSelGenres([])
    setSelDirectors([]); setMinRating(0); setWatchYears([]); setRewatchFilter('All')
  }

  const toggleYear = (y: number) =>
    setWatchYears(prev => prev.includes(y) ? prev.filter(x => x !== y) : [...prev, y])

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2">
        {[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full" style={{background:'var(--blue)',opacity:0.4,animation:`pulse-dot 1.2s ease ${i*0.2}s infinite`}} />)}
      </div>
    </div>
  )

  const inputStyle = {
    background:'var(--fill)', border:'1px solid var(--fill-border)',
    borderRadius:'12px', padding:'8px 10px', fontFamily:'inherit',
    fontSize:'0.75rem', color:'var(--text)', width:'100%', outline:'none'
  }
  const labelStyle = {
    display:'block', fontFamily:'inherit', fontSize:'0.6rem', fontWeight:600,
    letterSpacing:'0.12em', textTransform:'uppercase' as const,
    color:'var(--sub)', marginBottom:'6px'
  }

  return (
    <div className="min-h-screen mesh-bg flex flex-col">
      {/* NAV + FILTER — single unified glass surface */}
      <div className="liquid-nav sticky top-0 z-40 border-b border-black/7 flex-shrink-0">
        <div className="max-w-[1200px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <Link href="/" className="flex items-center gap-2 hover:opacity-70 transition-opacity">
              <div style={{width:'22px',height:'22px',borderRadius:'6px',background:'#0071e3',display:'inline-flex',alignItems:'center',justifyContent:'center',fontFamily:'Georgia,serif',fontSize:'12px',fontWeight:300,color:'white',letterSpacing:'-0.5px',flexShrink:0}}>fc</div>
              <span className="font-display text-[1rem] font-light text-[var(--text)] hidden sm:inline">Film Collection</span>
            </Link>
            <span className="text-[var(--border)] hidden sm:inline">/</span>
            <span className="font-body text-[0.75rem] font-semibold text-[var(--sub)]">Stats</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAboutOpen(true)}
              className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors"
            >
              About
            </button>
            <ThemeToggle />
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl font-body text-[0.7rem] font-medium transition-all"
              style={{
                background: sidebarOpen ? 'rgba(0,113,227,0.08)' : 'var(--fill)',
                color: sidebarOpen ? 'var(--blue)' : 'var(--sub)',
                border: `1px solid ${sidebarOpen ? 'rgba(0,113,227,0.2)' : 'var(--fill-border)'}`,
              }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
              </svg>
              <span className="hidden sm:inline">Filters</span>
              {activeFilters > 0 && (
                <span className="w-4 h-4 rounded-full text-[0.6rem] font-medium text-white flex items-center justify-center" style={{background:'var(--blue)'}}>{activeFilters}</span>
              )}
            </button>
          </div>
        </div>

        {/* FILTER PANEL — rolls down inside the same glass surface */}
        <div style={{
          display:'grid',
          gridTemplateRows: sidebarOpen ? '1fr' : '0fr',
          transition:'grid-template-rows 0.28s cubic-bezier(0.4,0,0.2,1)',
        }}>
          <div style={{overflow:'hidden'}}>
            <div style={{borderTop:'1px solid var(--separator)'}}>
              <div className="max-w-[1200px] mx-auto px-4 sm:px-8 py-4 space-y-3">

            {/* Row 1: Searchable filters */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              {/* Film */}
              <div ref={filmRef} style={{position:'relative'}}>
                <label style={labelStyle}>Film</label>
                {selectedFilms.length > 0 && (
                  <div style={{display:'flex',flexWrap:'wrap',gap:'4px',marginBottom:'6px'}}>
                    {selectedFilms.map(f => (
                      <span key={f} style={{
                        display:'inline-flex',alignItems:'center',gap:'4px',
                        padding:'3px 8px',borderRadius:'100px',
                        background:'rgba(0,113,227,0.1)',color:'#0071e3',
                        fontSize:'0.7rem',fontFamily:'inherit',
                      }}>
                        {f}
                        <button onClick={() => setSelectedFilms(prev => prev.filter(x => x !== f))}
                          style={{background:'none',border:'none',cursor:'pointer',color:'#0071e3',fontSize:'0.85rem',lineHeight:1,padding:0,opacity:0.7}}>
                          ×
                        </button>
                      </span>
                    ))}
                  </div>
                )}
                <input
                  value={filmQuery}
                  onChange={e => { setFilmQuery(e.target.value); setFilmDropOpen(true) }}
                  onFocus={() => setFilmDropOpen(true)}
                  placeholder={selectedFilms.length > 0 ? 'Add more films…' : 'Search film title…'}
                  style={inputStyle}
                />
                {filmDropOpen && filmSuggestions.length > 0 && (
                  <div style={{
                    position:'absolute', left:0, right:0, top:'calc(100% + 4px)',
                    background:'var(--surface)', border:'1px solid var(--fill-border)',
                    borderRadius:'12px', boxShadow:'0 8px 24px rgba(0,0,0,0.1)',
                    zIndex:200, overflow:'hidden', maxHeight:'220px', overflowY:'auto',
                  }}>
                    {filmSuggestions.map(m => (
                      <div key={m.name}
                        onClick={() => { setSelectedFilms(prev => [...prev, m.name]); setFilmQuery(''); setFilmDropOpen(false) }}
                        style={{padding:'9px 14px',fontSize:'0.85rem',cursor:'pointer',fontFamily:'inherit',color:'var(--text)',borderBottom:'1px solid var(--separator)'}}
                        onMouseEnter={e => (e.currentTarget.style.background='var(--fill)')}
                        onMouseLeave={e => (e.currentTarget.style.background='var(--surface)')}
                      >
                        <span style={{fontWeight:500}}>{m.name}</span>
                        <span style={{color:'var(--muted)',marginLeft:'6px',fontSize:'0.7rem'}}>{m.releaseYear} · {m.language}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <MultiSelect label="Language" options={languages} selected={selLanguages} onChange={setSelLanguages} placeholder="Search languages…" />
              <MultiSelect label="Genre" options={genres} selected={selGenres} onChange={setSelGenres} placeholder="Search genres…" />
              <MultiSelect label="Director" options={directors.slice(0,300)} selected={selDirectors} onChange={setSelDirectors} placeholder="Search directors…" />
            </div>

            {/* Row 2: Range, year chips, rewatch, count */}
            <div className="flex flex-wrap items-end gap-4 pt-1">
              {/* Min Rating */}
              <div style={{minWidth:'140px',flex:'1 1 140px'}}>
                <label style={labelStyle}>Min Rating — {minRating.toFixed(1)}</label>
                <input type="range" min="0" max="10" step="0.5" value={minRating}
                  onChange={e => setMinRating(parseFloat(e.target.value))}
                  className="w-full" style={{marginTop:'6px',accentColor:'var(--blue)'}} />
              </div>

              {/* Watch Year */}
              <div style={{flex:'2 1 200px'}}>
                <label style={labelStyle}>Watch Year</label>
                <div className="flex flex-wrap gap-1.5" style={{marginTop:'6px'}}>
                  {allYears.map(y => (
                    <button key={y} onClick={() => toggleYear(y)}
                      className="px-3 py-1 rounded-full font-body text-[0.7rem] font-medium transition-all"
                      style={{
                        background: watchYears.includes(y) ? 'var(--blue)' : 'var(--fill)',
                        color: watchYears.includes(y) ? 'white' : 'var(--sub)',
                        border: `1px solid ${watchYears.includes(y) ? 'var(--blue)' : 'var(--fill-border)'}`,
                      }}>
                      {y}
                    </button>
                  ))}
                </div>
              </div>

              {/* Rewatch */}
              <div>
                <label style={labelStyle}>View</label>
                <div className="flex gap-1.5" style={{marginTop:'6px'}}>
                  {['All','Rewatched','First watch'].map(opt => (
                    <button key={opt} onClick={() => setRewatchFilter(opt)}
                      className="px-3 py-1 rounded-full font-body text-[0.7rem] font-medium transition-all"
                      style={{
                        background: rewatchFilter===opt ? 'var(--blue)' : 'var(--fill)',
                        color: rewatchFilter===opt ? 'white' : 'var(--sub)',
                        border: `1px solid ${rewatchFilter===opt ? 'var(--blue)' : 'var(--fill-border)'}`,
                      }}>
                      {opt}
                    </button>
                  ))}
                </div>
              </div>

              {/* Count + Clear */}
              <div className="flex items-center gap-3 ml-auto pb-0.5">
                {activeFilters > 0 && (
                  <button onClick={resetFilters} className="font-body text-[0.6rem] text-[var(--blue)] hover:opacity-70">Clear all</button>
                )}
                <p className="font-body text-[0.7rem] text-[var(--muted)]">
                  <span style={{color:'var(--text)',fontWeight:600}}>{filtered.length}</span> of {allMovies.length}
                </p>
              </div>
            </div>

              </div>
            </div>
          </div>
        </div>
      </div>

      {/* MAIN */}
      <main className="flex-1">
        <div className="max-w-[1200px] mx-auto px-4 sm:px-8 py-8">
          <StatsContent movies={filtered} allEntries={allEntries} watchYears={watchYears} />
        </div>
      </main>

      {chatOpen && <ChatPanel movies={allMovies} onClose={() => setChatOpen(false)} />}
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
      <ScrollJump />

      {/* FLOATING CHAT */}
      {!chatOpen && (
        <button onClick={() => setChatOpen(true)}
          className="fixed bottom-8 right-8 z-50 w-12 h-12 rounded-full flex items-center justify-center text-white transition-all hover:opacity-90"
          style={{background:'var(--blue)',boxShadow:'0 4px 16px rgba(0,0,0,0.2)'}}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </button>
      )}
    </div>
  )
}
