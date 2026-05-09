'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import type { Movie } from '@/lib/movies'
import ChatPanel from '@/components/ChatPanel'
import Link from 'next/link'
import StatsContent from '@/components/StatsContent'
import MultiSelect from '@/components/MultiSelect'
import FeedbackWidget from '@/components/FeedbackWidget'
import AboutModal from '@/components/AboutModal'

export default function StatsPage() {
  const [allMovies,  setAllMovies]  = useState<Movie[]>([])
  const [allEntries, setAllEntries] = useState<Movie[]>([])
  const [filtered,   setFiltered]   = useState<Movie[]>([])
  const [loading,    setLoading]    = useState(true)
  const [chatOpen,   setChatOpen]   = useState(false)
  const [sidebarOpen,setSidebarOpen]= useState(false)
  const [aboutOpen,  setAboutOpen]  = useState(false)

  // Filters
  const [selectedFilm,  setSelectedFilm]  = useState<string>('')
  const [filmQuery,     setFilmQuery]     = useState('')
  const [filmDropOpen,  setFilmDropOpen]  = useState(false)
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

  // Close film dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (filmRef.current && !filmRef.current.contains(e.target as Node)) setFilmDropOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const applyFilters = useCallback(() => {
    let f = [...allMovies]
    if (selectedFilm)         f = f.filter(m => m.name === selectedFilm)
    if (selLanguages.length)  f = f.filter(m => selLanguages.includes(m.language))
    if (selGenres.length)     f = f.filter(m => selGenres.some(g => m.genre.includes(g)))
    if (selDirectors.length)  f = f.filter(m => selDirectors.some(d => m.director.includes(d)))
    if (minRating > 0)        f = f.filter(m => m.tmdbRating >= minRating)
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
  }, [allMovies, allEntries, selectedFilm, selLanguages, selGenres, selDirectors, minRating, watchYears, rewatchFilter])

  useEffect(() => { applyFilters() }, [applyFilters])
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
  const languages = ['All', ...Array.from(new Set(allMovies.map(m => m.language))).sort()]
  const genres    = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()]
  const directors = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.director.split(',').map(d => d.trim()).filter(d => d && d !== 'N/A')))).sort()]
  const allYears  = Array.from(new Set(allEntries.map(e => parseInt('20' + e.date.split('/')[2])).filter(y => !isNaN(y)))).sort()

  // Film autocomplete
  const filmSuggestions = filmQuery.trim().length >= 1
    ? allMovies.filter(m => m.name.toLowerCase().includes(filmQuery.toLowerCase())).slice(0, 10)
    : []

  const activeFilters = [selectedFilm, selLanguages.length>0, selGenres.length>0, selDirectors.length>0, minRating>0, watchYears.length>0, rewatchFilter!=='All'].filter(Boolean).length

  const resetFilters = () => {
    setSelectedFilm(''); setFilmQuery('')
    setSelLanguages([]); setSelGenres([])
    setSelDirectors([]); setMinRating(0); setWatchYears([]); setRewatchFilter('All')
  }

  const toggleYear = (y: number) =>
    setWatchYears(prev => prev.includes(y) ? prev.filter(x => x !== y) : [...prev, y])

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2">
        {[0,1,2].map(i => <div key={i} className="w-2 h-2 rounded-full bg-blue-300" style={{animation:`pulse-dot 1.2s ease ${i*0.2}s infinite`}} />)}
      </div>
    </div>
  )

  const inputStyle = {
    background:'rgba(0,0,0,0.04)', border:'1px solid rgba(0,0,0,0.08)',
    borderRadius:'10px', padding:'8px 10px', fontFamily:'inherit',
    fontSize:'0.78rem', color:'var(--text)', width:'100%', outline:'none'
  }
  const labelStyle = {
    display:'block', fontFamily:'inherit', fontSize:'0.58rem', fontWeight:600,
    letterSpacing:'0.12em', textTransform:'uppercase' as const,
    color:'var(--sub)', marginBottom:'6px'
  }

  return (
    <div className="min-h-screen mesh-bg flex flex-col">
      {/* NAV */}
      <nav className="sticky top-0 z-40 border-b border-black/7 flex-shrink-0"
        style={{background:'rgba(245,245,247,0.85)',backdropFilter:'blur(20px)'}}>
        <div className="px-4 sm:px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <Link href="/" className="flex items-center gap-2 hover:opacity-70 transition-opacity">
              <div style={{width:'22px',height:'22px',borderRadius:'5px',background:'#0071e3',display:'inline-flex',alignItems:'center',justifyContent:'center',fontFamily:'Georgia,serif',fontSize:'12px',fontWeight:300,color:'white',letterSpacing:'-0.5px',flexShrink:0}}>fc</div>
              <span className="font-display text-lg font-light text-[var(--text)] hidden sm:inline">Film Collection</span>
            </Link>
            <span className="text-black/20 hidden sm:inline">/</span>
            <span className="font-body text-[0.75rem] font-semibold text-[var(--sub)]">Stats</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAboutOpen(true)}
              className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors"
            >
              About
            </button>
            <button onClick={() => setSidebarOpen(!sidebarOpen)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg font-body text-[0.72rem] font-medium transition-all"
              style={{
                background: sidebarOpen ? 'rgba(0,113,227,0.08)' : 'rgba(0,0,0,0.04)',
                color: sidebarOpen ? 'var(--blue)' : 'var(--sub)',
                border: `1px solid ${sidebarOpen ? 'rgba(0,113,227,0.2)' : 'rgba(0,0,0,0.08)'}`,
              }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>
              </svg>
              <span className="hidden sm:inline">Filters</span>
              {activeFilters > 0 && (
                <span className="w-4 h-4 rounded-full text-[0.58rem] font-bold bg-blue-500 text-white flex items-center justify-center">{activeFilters}</span>
              )}
            </button>
          </div>
        </div>
      </nav>

      {/* BODY */}
      <div className="flex flex-1 overflow-hidden">
        {/* SIDEBAR */}
        {sidebarOpen && (
          <>
            {/* Mobile backdrop */}
            <div className="fixed inset-0 bg-black/20 z-20 sm:hidden" onClick={() => setSidebarOpen(false)} />
            <aside className="fixed left-0 z-30 border-r border-black/7 overflow-y-auto"
              style={{
                top: '56px', bottom: 0,
                width: '260px',
                background: 'rgba(255,255,255,0.97)',
                backdropFilter: 'blur(24px)',
              }}>
              <div className="p-5">
                <div className="flex items-center justify-between mb-5">
                  <p style={{...labelStyle, marginBottom:0}}>Refine</p>
                  <div className="flex items-center gap-3">
                    {activeFilters > 0 && (
                      <button onClick={resetFilters} className="font-body text-[0.65rem] text-[var(--blue)] hover:opacity-70">Clear all</button>
                    )}
                    <button onClick={() => setSidebarOpen(false)} className="text-[var(--muted)] hover:text-[var(--text)]">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                      </svg>
                    </button>
                  </div>
                </div>

                <div className="space-y-5">
                  {/* Film autocomplete */}
                  <div ref={filmRef}>
                    <label style={labelStyle}>Film</label>
                    <div style={{position:'relative'}}>
                      <input
                        value={selectedFilm || filmQuery}
                        onChange={e => {
                          setFilmQuery(e.target.value)
                          setSelectedFilm('')
                          setFilmDropOpen(true)
                        }}
                        onFocus={() => setFilmDropOpen(true)}
                        placeholder="Search film title…"
                        style={inputStyle}
                      />
                      {selectedFilm && (
                        <button
                          onClick={() => { setSelectedFilm(''); setFilmQuery('') }}
                          style={{
                            position:'absolute', right:'8px', top:'50%', transform:'translateY(-50%)',
                            background:'none', border:'none', cursor:'pointer',
                            color:'var(--muted)', fontSize:'16px', lineHeight:1,
                          }}
                        >×</button>
                      )}
                    </div>
                    {filmDropOpen && filmSuggestions.length > 0 && !selectedFilm && (
                      <div style={{
                        position:'absolute', left:'20px', right:'20px',
                        background:'white', border:'1px solid rgba(0,0,0,0.1)',
                        borderRadius:'12px', boxShadow:'0 8px 24px rgba(0,0,0,0.1)',
                        zIndex:100, overflow:'hidden', maxHeight:'220px', overflowY:'auto',
                      }}>
                        {filmSuggestions.map(m => (
                          <div
                            key={m.name}
                            onClick={() => {
                              setSelectedFilm(m.name)
                              setFilmQuery(m.name)
                              setFilmDropOpen(false)
                            }}
                            style={{
                              padding:'9px 14px', fontSize:'0.8rem', cursor:'pointer',
                              fontFamily:'inherit', color:'var(--text)',
                              borderBottom:'1px solid rgba(0,0,0,0.04)',
                            }}
                            onMouseEnter={e => (e.currentTarget.style.background='rgba(0,0,0,0.03)')}
                            onMouseLeave={e => (e.currentTarget.style.background='white')}
                          >
                            <span style={{fontWeight:500}}>{m.name}</span>
                            <span style={{color:'var(--muted)',marginLeft:'6px',fontSize:'0.72rem'}}>{m.releaseYear} · {m.language}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Language */}
                  <MultiSelect label="Language" options={languages.filter(l => l !== 'All')} selected={selLanguages} onChange={setSelLanguages} placeholder="Search languages…" />

                  {/* Genre */}
                  <MultiSelect label="Genre" options={genres.filter(g => g !== 'All')} selected={selGenres} onChange={setSelGenres} placeholder="Search genres…" />

                  {/* Director */}
                  <MultiSelect label="Director" options={directors.filter(d => d !== 'All').slice(0,300)} selected={selDirectors} onChange={setSelDirectors} placeholder="Search directors…" />

                  {/* Min Rating */}
                  <div>
                    <label style={labelStyle}>Min Rating — {minRating.toFixed(1)}</label>
                    <input type="range" min="0" max="10" step="0.5" value={minRating}
                      onChange={e => setMinRating(parseFloat(e.target.value))}
                      className="w-full accent-blue-500" />
                    <div className="flex justify-between font-body text-[0.58rem] text-[var(--muted)] mt-1">
                      <span>0</span><span>10</span>
                    </div>
                  </div>

                  {/* Watch Year */}
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <label style={{...labelStyle, marginBottom:0}}>Watch Year</label>
                      {watchYears.length > 0 && (
                        <button onClick={() => setWatchYears([])} className="font-body text-[0.6rem] text-[var(--blue)]">Clear</button>
                      )}
                    </div>
                    <div className="grid grid-cols-3 gap-1.5">
                      {allYears.map(y => (
                        <button key={y} onClick={() => toggleYear(y)}
                          className="py-1.5 rounded-lg font-body text-[0.7rem] font-medium transition-all"
                          style={{
                            background: watchYears.includes(y) ? 'var(--blue)' : 'rgba(0,0,0,0.04)',
                            color: watchYears.includes(y) ? 'white' : 'var(--sub)',
                            border: `1px solid ${watchYears.includes(y) ? 'var(--blue)' : 'rgba(0,0,0,0.08)'}`,
                          }}>
                          {y}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Rewatch */}
                  <div>
                    <label style={labelStyle}>View</label>
                    <div className="flex flex-col gap-1.5">
                      {['All','Rewatched','First watch'].map(opt => (
                        <button key={opt} onClick={() => setRewatchFilter(opt)}
                          className="py-1.5 rounded-lg font-body text-[0.72rem] font-medium text-left px-3 transition-all"
                          style={{
                            background: rewatchFilter===opt ? 'var(--blue)' : 'rgba(0,0,0,0.04)',
                            color: rewatchFilter===opt ? 'white' : 'var(--sub)',
                            border: `1px solid ${rewatchFilter===opt ? 'var(--blue)' : 'rgba(0,0,0,0.08)'}`,
                          }}>
                          {opt}
                        </button>
                      ))}
                    </div>
                  </div>

                  <div className="pt-3 border-t border-black/7">
                    <p className="font-body text-[0.7rem] text-[var(--muted)]">
                      <span style={{color:'var(--text)',fontWeight:600}}>{filtered.length}</span> of {allMovies.length} films
                    </p>
                  </div>
                </div>
              </div>
            </aside>
          </>
        )}

        {/* MAIN */}
        <main className="flex-1 overflow-y-auto transition-all duration-300"
          style={{marginLeft: sidebarOpen ? '260px' : '0'}}>
          <div className="max-w-[1200px] mx-auto px-4 sm:px-8 py-8">
            <StatsContent movies={filtered} allEntries={allEntries} watchYears={watchYears} />
          </div>
        </main>
      </div>

      {chatOpen && <ChatPanel movies={allMovies} onClose={() => setChatOpen(false)} />}
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
      <FeedbackWidget />

      {/* FLOATING CHAT */}
      <button onClick={() => setChatOpen(true)}
        className="fixed bottom-8 right-8 z-50 w-14 h-14 rounded-full flex items-center justify-center text-white shadow-2xl transition-all hover:scale-105"
        style={{background:'linear-gradient(135deg,#0071e3,#34aadc)',boxShadow:'0 8px 32px rgba(0,113,227,0.4)'}}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>
    </div>
  )
}
