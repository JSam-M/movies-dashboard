'use client'

import { useState, useEffect, useRef } from 'react'
import type { Movie } from '@/lib/movies'
import ChatPanel from '@/components/ChatPanel'
import MultiSelect from '@/components/MultiSelect'
import Link from 'next/link'

type SortKey = 'rating' | 'rewatched' | 'date'
type SortDir = 'desc' | 'asc'

// Daily rotation — seeded by date so same 6 films show all day
function getDailyPicks(movies: Movie[]): Movie[] {
  const today = new Date()
  const seed  = today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate()
  const eligible = movies.filter(m => m.tmdbRating >= 7.0)
  const shuffled = [...eligible].sort((a, b) => {
    const ha = ((seed * 1103515245 + (a.name.charCodeAt(0) || 0)) >>> 0) % 1000
    const hb = ((seed * 1103515245 + (b.name.charCodeAt(0) || 0)) >>> 0) % 1000
    return ha - hb
  })
  // Mix: top 3 by rating from shuffled, top 2 rewatched, 1 random
  const topRated    = [...shuffled].sort((a,b) => b.tmdbRating - a.tmdbRating).slice(0,3)
  const topRewatched = shuffled.filter(m => m.timesWatched >= 2).sort((a,b) => b.timesWatched - a.timesWatched).slice(0,2)
  const rest        = shuffled.filter(m => !topRated.find(x=>x.name===m.name) && !topRewatched.find(x=>x.name===m.name))
  return Array.from(new Map([...topRated, ...topRewatched, ...rest].map(m=>[m.name,m])).values()).slice(0,6)
}

function MovieModal({ movie, onClose }: { movie: Movie; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-8"
      style={{background:'rgba(0,0,0,0.25)',backdropFilter:'blur(12px)'}}
      onClick={onClose}>
      <div className="relative w-full animate-fade-up"
        style={{maxWidth:'480px',background:'rgba(255,255,255,0.96)',borderRadius:'24px',padding:'32px',boxShadow:'0 32px 80px rgba(0,0,0,0.18),0 8px 24px rgba(0,0,0,0.08)',border:'1px solid rgba(255,255,255,0.9)'}}
        onClick={e => e.stopPropagation()}>
        <button onClick={onClose}
          className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--text)] transition-colors">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>

        <div className="flex items-start justify-between mb-4 pr-8">
          <div>
            <p className="font-body text-[0.62rem] font-semibold tracking-[0.1em] uppercase px-2 py-1 rounded-full mb-3 inline-block"
              style={{background:'rgba(0,113,227,0.07)',color:'var(--blue)'}}>
              {movie.genre.split(',')[0].trim()}
            </p>
            <h2 className="font-display text-[1.6rem] font-light text-[var(--text)] leading-tight">{movie.name}</h2>
            <p className="font-body text-[0.75rem] text-[var(--sub)] mt-1">
              {movie.releaseYear} · {movie.language} · {movie.runtime}
              {movie.timesWatched >= 2 && <span className="text-amber-500 ml-2">★ Watched {movie.timesWatched}×</span>}
            </p>
          </div>
          <div className="text-right flex-shrink-0 ml-4">
            <div className="font-display text-[2.2rem] font-light" style={{color:'var(--blue)'}}>{movie.tmdbRating.toFixed(1)}</div>
            <div className="font-body text-[0.6rem] text-[var(--muted)]">TMDb</div>
          </div>
        </div>

        <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed mb-5">{movie.overview || 'No overview available.'}</p>

        <div className="flex flex-wrap gap-2 pt-4 border-t border-black/7">
          <span className="font-body text-[0.72rem] text-[var(--sub)]">Director:</span>
          <span className="font-body text-[0.72rem] text-[var(--text)]">{movie.director.split(',')[0].trim()}</span>
          {movie.genre.split(',').slice(0,3).map(g => (
            <span key={g} className="font-body text-[0.65rem] px-2 py-0.5 rounded-full"
              style={{background:'rgba(0,0,0,0.05)',color:'var(--sub)'}}>{g.trim()}</span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function DiscoverPage() {
  const [allMovies,   setAllMovies]   = useState<Movie[]>([])
  const [filtered,    setFiltered]    = useState<Movie[]>([])
  const [loading,     setLoading]     = useState(true)
  const [chatOpen,    setChatOpen]    = useState(false)
  const [aiQuery,     setAiQuery]     = useState('')
  const [initialMsg,  setInitialMsg]  = useState('')
  const [search,      setSearch]      = useState('')
  const [genres,      setGenres]      = useState<string[]>([])
  const [languages,   setLanguages]   = useState<string[]>([])
  const [showRewatched,setShowRewatched]=useState(false)
  const [sortKey,     setSortKey]     = useState<SortKey>('rating')
  const [sortDir,     setSortDir]     = useState<SortDir>('desc')
  const [stats,       setStats]       = useState<Record<string,unknown>>({})
  const [selectedMovie,setSelectedMovie]=useState<Movie|null>(null)
  const [dailyPicks,  setDailyPicks]  = useState<Movie[]>([])
  const aiInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, stats: s }) => {
        setAllMovies(movies); setStats(s)
        setDailyPicks(getDailyPicks(movies))
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    let f = [...allMovies]
    if (search)           f = f.filter(m => m.name.toLowerCase().includes(search.toLowerCase()) || m.director.toLowerCase().includes(search.toLowerCase()))
    if (genres.length)    f = f.filter(m => genres.some(g => m.genre.includes(g)))
    if (languages.length) f = f.filter(m => languages.includes(m.language))
    if (showRewatched)    f = f.filter(m => m.timesWatched >= 2)
    f = [...f].sort((a, b) => {
      let diff = 0
      if (sortKey === 'rating')    diff = a.tmdbRating - b.tmdbRating
      if (sortKey === 'rewatched') diff = a.timesWatched - b.timesWatched
      if (sortKey === 'date')      diff = a.date.localeCompare(b.date)
      return sortDir === 'desc' ? -diff : diff
    })
    setFiltered(f)
  }, [search, genres, languages, showRewatched, sortKey, sortDir, allMovies])

  const handleAiSearch = () => {
    if (!aiQuery.trim()) return
    setInitialMsg(aiQuery.trim()); setAiQuery(''); setChatOpen(true)
  }

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const sortArrow = (key: SortKey) => sortKey === key ? (sortDir === 'desc' ? ' ↓' : ' ↑') : ''

  const allGenreOpts = Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()
  const allLangOpts  = Array.from(new Set(allMovies.map(m => m.language))).sort()

  const btnStyle = (active: boolean): React.CSSProperties => ({
    padding:'8px 14px', borderRadius:'100px',
    border:`1px solid ${active?'#0071e3':'rgba(0,0,0,0.08)'}`,
    background: active ? '#0071e3' : 'white',
    color: active ? 'white' : 'var(--sub)',
    fontSize:'0.78rem', fontFamily:'inherit', cursor:'pointer',
    fontWeight: active ? 500 : 400, transition:'all 0.15s',
  })

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2">
        {[0,1,2].map(i=><div key={i} className="w-2 h-2 rounded-full bg-blue-300" style={{animation:`pulse-dot 1.2s ease ${i*0.2}s infinite`}}/>)}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen mesh-bg">
      {/* NAV */}
      <nav className="sticky top-0 z-40 border-b border-black/7" style={{background:'rgba(245,245,247,0.85)',backdropFilter:'blur(20px)'}}>
        <div className="max-w-[1200px] mx-auto px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div style={{width:'22px',height:'22px',borderRadius:'5px',background:'#0071e3',display:'inline-flex',alignItems:'center',justifyContent:'center',fontFamily:'Georgia,serif',fontSize:'12px',fontWeight:300,color:'white',letterSpacing:'-0.5px',flexShrink:0}}>fc</div>
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
          <p className="font-body text-[1rem] text-[var(--sub)] max-w-md mx-auto leading-relaxed mb-8">
            {stats.total as number} films watched. Not sure what to watch? The AI knows this collection inside out.
          </p>

          {/* AI search pill */}
          <div style={{position:'relative',maxWidth:'520px',margin:'0 auto'}}>
            <div style={{display:'flex',alignItems:'center',background:'#0071e3',borderRadius:'100px',padding:'6px 6px 6px 18px',boxShadow:'0 8px 32px rgba(0,113,227,0.35)'}}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="2" style={{flexShrink:0,marginRight:'10px'}}>
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              <input ref={aiInputRef} value={aiQuery} onChange={e=>setAiQuery(e.target.value)}
                onKeyDown={e=>e.key==='Enter'&&handleAiSearch()}
                placeholder="What should I watch tonight?"
                style={{flex:1,background:'transparent',border:'none',outline:'none',color:'white',fontSize:'0.95rem',fontFamily:'inherit'}}
                onFocus={e => e.target.style.opacity='1'}
              />
              <style>{`input::placeholder { color: rgba(255,255,255,0.75); }`}</style>
              <button onClick={handleAiSearch} style={{width:'38px',height:'38px',borderRadius:'100px',flexShrink:0,background:'rgba(255,255,255,0.2)',border:'none',cursor:'pointer',display:'flex',alignItems:'center',justifyContent:'center'}}
                onMouseEnter={e=>(e.currentTarget.style.background='rgba(255,255,255,0.3)')}
                onMouseLeave={e=>(e.currentTarget.style.background='rgba(255,255,255,0.2)')}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            </div>
          </div>

          {/* Quick prompts */}
          <div style={{display:'flex',flexWrap:'wrap',gap:'8px',justifyContent:'center',marginTop:'14px'}}>
            {['Something feel-good','I loved Parasite — suggest similar','Best Tamil films','Under 2 hours','Hidden gems','Watch with family'].map(q=>(
              <button key={q} onClick={()=>{setInitialMsg(q);setChatOpen(true)}} style={{padding:'6px 14px',borderRadius:'100px',border:'1px solid rgba(0,0,0,0.1)',background:'white',fontSize:'0.75rem',fontFamily:'inherit',color:'var(--sub)',cursor:'pointer'}}>{q}</button>
            ))}
          </div>
          <p className="font-body text-[0.7rem] text-[var(--muted)] mt-3">AI recommends only from films actually watched</p>
        </div>

        {/* TOP PICKS — daily rotation */}
        <div className="mb-16">
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Today&apos;s Picks</p>
              <p className="font-display text-[2rem] font-light text-[var(--text)]">Featured Films</p>
            </div>
            <p className="font-body text-[0.72rem] text-[var(--muted)]">Refreshes daily · Click for details</p>
          </div>
          <div className="grid grid-cols-3 gap-4">
            {dailyPicks.map(m=>(
              <button key={m.name} onClick={()=>setSelectedMovie(m)} className="glass rounded-2xl p-5 text-left hover:shadow-lg transition-all hover:scale-[1.01]" style={{cursor:'pointer'}}>
                <div className="flex items-start justify-between mb-3">
                  <span className="font-body text-[0.62rem] font-semibold tracking-[0.08em] uppercase px-2 py-1 rounded-full"
                    style={{background:'rgba(0,113,227,0.07)',color:'var(--blue)'}}>
                    {m.genre.split(',')[0].trim()}
                  </span>
                  <div className="text-right">
                    <span className="font-display text-[1.3rem] font-light" style={{color:'var(--blue)'}}>{m.tmdbRating.toFixed(1)}</span>
                    {m.timesWatched>=2&&<span className="block font-body text-[0.6rem] text-amber-500 font-semibold">{m.timesWatched}× watched</span>}
                  </div>
                </div>
                <p className="font-display text-[1.1rem] font-light text-[var(--text)] leading-tight mb-1">{m.name}</p>
                <p className="font-body text-[0.7rem] text-[var(--sub)] mb-2">{m.releaseYear} · {m.language} · {m.runtime}</p>
                <p className="font-body text-[0.76rem] text-[var(--sub)] leading-relaxed line-clamp-2">{m.overview}</p>
                <p className="font-body text-[0.65rem] text-[var(--blue)] mt-3">Tap to read more →</p>
              </button>
            ))}
          </div>
        </div>

        {/* BROWSE */}
        <div>
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Browse</p>
              <p className="font-display text-[2rem] font-light text-[var(--text)]">Collection</p>
            </div>
            <p className="font-body text-[0.75rem] text-[var(--muted)]">{filtered.length} films</p>
          </div>

          {/* Filters — with proper z-index stacking */}
          <div className="mb-6 space-y-4" style={{position:'relative',zIndex:20}}>
            {/* Sort + favourites row */}
            <div className="flex gap-3 flex-wrap items-center">
              <div className="relative flex-1 min-w-[200px]">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#86868b" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search films or directors…"
                  className="w-full pl-9 pr-4 py-2.5 rounded-xl font-body text-sm outline-none"
                  style={{background:'white',border:'1px solid rgba(0,0,0,0.08)',color:'var(--text)'}}/>
              </div>
              <div className="flex gap-2">
                {(['rating','rewatched','date'] as SortKey[]).map(k=>(
                  <button key={k} onClick={()=>handleSort(k)} style={btnStyle(sortKey===k)}>
                    {k==='rating'?'Rating':k==='rewatched'?'Rewatched':'Date'}{sortArrow(k)}
                  </button>
                ))}
              </div>
              <button onClick={()=>setShowRewatched(!showRewatched)} style={btnStyle(showRewatched)}>★ Favourites</button>
            </div>

            {/* Genre + Language multiselect — z-index ensures dropdown appears above film list */}
            <div className="grid grid-cols-2 gap-4" style={{position:'relative',zIndex:30}}>
              <div className="glass rounded-xl p-4" style={{overflow:'visible'}}>
                <MultiSelect label="Genre" options={allGenreOpts} selected={genres} onChange={setGenres} placeholder="Search genres…"/>
              </div>
              <div className="glass rounded-xl p-4" style={{overflow:'visible'}}>
                <MultiSelect label="Language" options={allLangOpts} selected={languages} onChange={setLanguages} placeholder="Search languages…"/>
              </div>
            </div>
          </div>

          {/* Film list — z-index lower than filters */}
          <div className="grid grid-cols-1 gap-2" style={{position:'relative',zIndex:10}}>
            {filtered.slice(0,60).map(m=>(
              <button key={m.name} onClick={()=>setSelectedMovie(m)}
                className="glass rounded-xl px-5 py-3.5 flex items-center gap-5 hover:bg-white/90 transition-all text-left w-full">
                <div className="font-display text-[1.5rem] font-light w-12 text-center flex-shrink-0" style={{color:'var(--blue)'}}>
                  {m.tmdbRating>0?m.tmdbRating.toFixed(1):'—'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-body text-[0.88rem] font-medium text-[var(--text)] truncate">
                    {m.name}{m.timesWatched>=2&&<span className="text-amber-400 ml-1">★</span>}
                  </p>
                  <p className="font-body text-[0.7rem] text-[var(--sub)]">{m.releaseYear} · {m.director.split(',')[0].trim()} · {m.runtime}</p>
                </div>
                <div className="hidden md:flex gap-2 flex-shrink-0 items-center">
                  {m.timesWatched>=2&&<span className="font-body text-[0.62rem] px-2 py-1 rounded-full font-semibold" style={{background:'rgba(251,191,36,0.12)',color:'#d97706'}}>{m.timesWatched}×</span>}
                  <span className="font-body text-[0.62rem] px-2 py-1 rounded-full" style={{background:'rgba(0,0,0,0.04)',color:'var(--sub)'}}>{m.language}</span>
                  <span className="font-body text-[0.62rem] px-2 py-1 rounded-full" style={{background:'rgba(0,0,0,0.04)',color:'var(--sub)'}}>{m.genre.split(',')[0].trim()}</span>
                </div>
              </button>
            ))}

            {/* 60 film limit message */}
            {filtered.length > 60 && (
              <div className="glass rounded-xl px-5 py-5 text-center">
                <p className="font-body text-[0.82rem] text-[var(--text)] mb-1">
                  Showing 60 of {filtered.length} films
                </p>
                <p className="font-body text-[0.75rem] text-[var(--sub)] mb-3">
                  Refine filters above, or browse the complete list with full sorting and filters on the Stats page.
                </p>
                <Link href="/stats" className="font-body text-[0.78rem] font-semibold text-[var(--blue)] hover:opacity-70 transition-opacity">
                  View full collection on Stats →
                </Link>
              </div>
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
      <button onClick={()=>setChatOpen(true)}
        className="fixed bottom-8 right-8 z-50 w-14 h-14 rounded-full flex items-center justify-center text-white shadow-2xl transition-all hover:scale-105"
        style={{background:'linear-gradient(135deg,#0071e3,#34aadc)',boxShadow:'0 8px 32px rgba(0,113,227,0.4)'}}>
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </button>

      {/* Movie modal */}
      {selectedMovie && <MovieModal movie={selectedMovie} onClose={()=>setSelectedMovie(null)}/>}

      {chatOpen && <ChatPanel movies={allMovies} initialMessage={initialMsg} onClose={()=>{setChatOpen(false);setInitialMsg('')}}/>}
    </div>
  )
}
