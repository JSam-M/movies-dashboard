'use client'

import { useState, useEffect, useRef } from 'react'
import { track } from '@/lib/track'
import type { Movie } from '@/lib/movies'
import ChatPanel from '@/components/ChatPanel'
import MultiSelect from '@/components/MultiSelect'
import AboutModal from '@/components/AboutModal'
import ScrollJump from '@/components/ScrollJump'
import ThemeToggle from '@/components/ThemeToggle'
import Link from 'next/link'

type SortKey = 'rating' | 'rewatched' | 'date'
type SortDir = 'desc' | 'asc'

function getDailySeed(): number {
  const d = new Date()
  return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate()
}

function seededRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0
    return s / 0xFFFFFFFF
  }
}

function getDailyChips(movies: Movie[]): string[] {
  const rand = Math.random.bind(Math)
  const pick = <T,>(arr: T[]): T => arr[Math.floor(rand() * arr.length)]

  // Top directors (≥4 films)
  const directorCounts: Record<string, number> = {}
  movies.forEach(m => m.director.split(',').forEach(d => {
    const name = d.trim(); if (name) directorCounts[name] = (directorCounts[name] || 0) + 1
  }))
  const topDirectors = Object.entries(directorCounts).filter(([,c]) => c >= 4).map(([d]) => d)

  // Languages with enough films
  const langCounts: Record<string, number> = {}
  movies.forEach(m => { if (m.language) langCounts[m.language] = (langCounts[m.language] || 0) + 1 })
  const topLangs = Object.entries(langCounts).filter(([,c]) => c >= 10).map(([l]) => l)

  // Genres
  const genreCounts: Record<string, number> = {}
  movies.forEach(m => m.genre.split(',').forEach(g => {
    const name = g.trim(); if (name) genreCounts[name] = (genreCounts[name] || 0) + 1
  }))
  const topGenres = Object.entries(genreCounts).filter(([,c]) => c >= 8).map(([g]) => g)

  // Highly-rated films for "I loved X" chip
  const gems = movies.filter(m => m.tmdbRating >= 8.0)

  const moodPool = [
    'Something feel-good', 'Under 2 hours', 'Hidden gems',
    'Watch with family', 'A slow-burn thriller', 'Something thought-provoking',
    'A great one-liner film', 'Best of the decade',
  ]

  const chips: string[] = []
  if (topDirectors.length) chips.push(`Films by ${pick(topDirectors)}`)
  if (topLangs.length)     chips.push(`Best ${pick(topLangs)} films`)
  if (gems.length)         chips.push(`I loved ${pick(gems).name} — suggest similar`)
  if (topGenres.length)    chips.push(`Best ${pick(topGenres)} films`)

  // Fill remaining slots from mood pool (pick 2, no duplicates)
  const shuffledMood = [...moodPool].sort(() => rand() - 0.5)
  shuffledMood.slice(0, Math.max(2, 6 - chips.length)).forEach(m => chips.push(m))

  return chips.slice(0, 6)
}

function getDailyPicks(movies: Movie[]): Movie[] {
  const today = new Date()
  const seed  = today.getFullYear() * 10000 + (today.getMonth() + 1) * 100 + today.getDate()
  const eligible = movies.filter(m => m.tmdbRating >= 7.0)

  // Seeded hash using the full film name so every film gets a unique, day-varying position
  function filmHash(name: string): number {
    let h = seed
    for (let i = 0; i < name.length; i++) {
      h = Math.imul(h ^ name.charCodeAt(i), 2654435761)
      h ^= h >>> 16
    }
    return (h >>> 0)
  }

  const shuffled = [...eligible].sort((a, b) => filmHash(a.name) - filmHash(b.name))
  return shuffled.slice(0, 6)
}

function MovieModal({ movie, onClose }: { movie: Movie; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8"
      style={{background:'rgba(0,0,0,0.25)',backdropFilter:'blur(12px)'}}
      onClick={onClose}>
      <div className="relative w-full animate-fade-up rounded-3xl"
        style={{maxWidth:'480px',background:'var(--modal-bg)',padding:'28px',boxShadow:'0 32px 80px rgba(0,0,0,0.18)',border:'1px solid var(--glass-border)'}}
        onClick={e => e.stopPropagation()}>
        <button onClick={onClose} className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--text)] transition-colors">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
          </svg>
        </button>
        <div className="flex items-start justify-between mb-4 pr-8">
          <div>
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase px-2 py-1 rounded-full mb-3 inline-block"
              style={{background:'rgba(0,113,227,0.07)',color:'var(--blue)'}}>
              {movie.genre.split(',')[0].trim()}
            </p>
            <h2 className="font-display text-[1.5rem] font-light text-[var(--text)] leading-tight">{movie.name}</h2>
            <p className="font-body text-[0.75rem] text-[var(--sub)] mt-1">
              {movie.releaseYear} · {movie.language} · {movie.runtime}
              {movie.timesWatched >= 2 && <span style={{color:'var(--gold)'}} className="ml-2">★ Watched {movie.timesWatched}×</span>}
            </p>
          </div>
          <div className="text-right flex-shrink-0 ml-4">
            <div className="font-display text-[2.2rem] font-light" style={{color:'var(--blue)'}}>{movie.tmdbRating.toFixed(1)}</div>
            <div className="font-body text-[0.6rem] text-[var(--muted)]">IMDb</div>
          </div>
        </div>
        <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed mb-5">{movie.overview || 'No overview available.'}</p>
        <div className="flex flex-wrap gap-2 pt-4" style={{borderTop:'1px solid var(--separator)'}}>
          <span className="font-body text-[0.75rem] text-[var(--sub)]">Director:</span>
          <span className="font-body text-[0.75rem] text-[var(--text)]">{movie.director.split(',')[0].trim()}</span>
          {movie.genre.split(',').slice(0,3).map(g => (
            <span key={g} className="font-body text-[0.6rem] px-2 py-0.5 rounded-full"
              style={{background:'var(--fill)',color:'var(--sub)'}}>{g.trim()}</span>
          ))}
        </div>
      </div>
    </div>
  )
}

export default function DiscoverPage() {
  const [allMovies,    setAllMovies]    = useState<Movie[]>([])
  const [filtered,     setFiltered]     = useState<Movie[]>([])
  const [loading,      setLoading]      = useState(true)
  const [chatOpen,     setChatOpen]     = useState(false)
  const [aboutOpen,    setAboutOpen]    = useState(false)
  const [aiQuery,      setAiQuery]      = useState('')
  const [initialMsg,   setInitialMsg]   = useState('')
  const [search,       setSearch]       = useState('')
  const [genres,       setGenres]       = useState<string[]>([])
  const [languages,    setLanguages]    = useState<string[]>([])
  const [showRewatched,setShowRewatched]= useState(false)
  const [sortKey,      setSortKey]      = useState<SortKey>('rating')
  const [sortDir,      setSortDir]      = useState<SortDir>('desc')
  const [stats,        setStats]        = useState<Record<string,unknown>>({})
  const [selectedMovie,setSelectedMovie]= useState<Movie|null>(null)
  const [dailyPicks,   setDailyPicks]   = useState<Movie[]>([])
  const [quickPrompts, setQuickPrompts] = useState<string[]>([])
  const aiInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => { track('page_view', '/') }, [])

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, stats: s }) => {
        setAllMovies(movies); setStats(s)
        setDailyPicks(getDailyPicks(movies))
        setQuickPrompts(getDailyChips(movies))
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
    padding:'8px 14px', borderRadius:'9999px',
    border:`1px solid ${active?'var(--blue)':'var(--fill-border)'}`,
    background: active ? 'var(--blue)' : 'var(--surface)',
    color: active ? 'white' : 'var(--sub)',
    fontSize:'0.75rem', fontFamily:'inherit', cursor:'pointer',
    fontWeight: active ? 500 : 400, transition:'all 0.15s',
  })

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="flex gap-2">
        {[0,1,2].map(i=><div key={i} className="w-2 h-2 rounded-full" style={{background:'var(--blue)',opacity:0.4,animation:`pulse-dot 1.2s ease ${i*0.2}s infinite`}}/>)}
      </div>
    </div>
  )

  return (
    <div className="min-h-screen mesh-bg">
      {/* NAV */}
      <nav className="liquid-nav sticky top-0 z-40 border-b border-black/7">
        <div className="max-w-[1200px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div style={{width:'22px',height:'22px',borderRadius:'6px',background:'#0071e3',display:'inline-flex',alignItems:'center',justifyContent:'center',fontFamily:'Georgia,serif',fontSize:'12px',fontWeight:300,color:'white',letterSpacing:'-0.5px',flexShrink:0}}>fc</div>
            <span className="font-display text-[1rem] font-light text-[var(--text)] hidden sm:inline">Film Collection</span>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAboutOpen(true)}
              className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors"
            >
              About
            </button>
            <Link href="/stats" className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors flex items-center gap-1.5">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
              </svg>
              <span className="hidden sm:inline">My Stats</span>
            </Link>
            <ThemeToggle />
          </div>
        </div>
      </nav>

      <div className="max-w-[1200px] mx-auto px-4 sm:px-8 py-12 sm:py-20">

        {/* HERO */}
        <div className="text-center mb-4 sm:mb-6">
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.2em] uppercase text-[var(--sub)] mb-5">Personal Film Archive · Since 2019</p>
          <h1 className="font-display text-[clamp(2.8rem,7vw,6.5rem)] font-light leading-[0.9] tracking-tight text-[var(--text)] mb-8">
            A life in{' '}
            <em style={{fontStyle:'italic',background:'linear-gradient(135deg,#0071e3,#34aadc)',WebkitBackgroundClip:'text',WebkitTextFillColor:'transparent'}}>cinema</em>
          </h1>
          <p className="font-body text-[1rem] text-[var(--sub)] max-w-md mx-auto leading-relaxed mb-8">
            {stats.total as number} films watched. Not sure what to watch? The AI knows this collection inside out.
          </p>

          {/* AI search pill */}
          <div style={{position:'relative',maxWidth:'520px',margin:'0 auto'}}>
            <div style={{display:'flex',alignItems:'center',background:'#0071e3',borderRadius:'9999px',padding:'6px 6px 6px 18px',boxShadow:'0 8px 32px rgba(0,113,227,0.35)'}}>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.8)" strokeWidth="2" style={{flexShrink:0,marginRight:'10px'}}>
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
              </svg>
              <input ref={aiInputRef} value={aiQuery} onChange={e=>setAiQuery(e.target.value)}
                onKeyDown={e=>e.key==='Enter'&&handleAiSearch()}
                placeholder="What should I watch tonight?"
                className="hero-search"
                style={{flex:1,background:'transparent',border:'none',outline:'none',color:'white',fontSize:'1rem',fontFamily:'inherit',colorScheme:'light'}}
              />
              <button onClick={handleAiSearch} style={{width:'38px',height:'38px',borderRadius:'9999px',flexShrink:0,background:'rgba(255,255,255,0.2)',border:'none',display:'flex',alignItems:'center',justifyContent:'center'}}
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
            {quickPrompts.map(q=>(
              <button key={q} onClick={()=>{setInitialMsg(q);setChatOpen(true)}}
                style={{padding:'6px 14px',borderRadius:'9999px',border:'1px solid var(--fill-border)',background:'var(--glass)',backdropFilter:'blur(12px)',fontSize:'0.75rem',fontFamily:'inherit',color:'var(--sub)',cursor:'pointer'}}>
                {q}
              </button>
            ))}
          </div>
          <p className="font-body text-[0.7rem] text-[var(--muted)] mt-3">AI recommends only from films actually watched</p>
        </div>

        {/* DAILY PICKS */}
        <div className="mb-12 sm:mb-16">
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-display text-[1.5rem] sm:text-[2rem] font-light text-[var(--text)]">Today&apos;s Picks</p>
            </div>
            <p className="font-body text-[0.7rem] text-[var(--muted)] hidden sm:block">Refreshes daily · Click for details</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {dailyPicks.map(m=>(
              <button key={m.name} onClick={()=>setSelectedMovie(m)} className="glass rounded-2xl p-5 text-left transition-all hover:opacity-90">
                <div className="flex items-start justify-between mb-1">
                  <em className="font-display text-[0.85rem] font-light" style={{fontStyle:'italic',color:'var(--sub)'}}>
                    {m.genre.split(',')[0].trim()}
                  </em>
                  <span className="font-display text-[1.2rem] font-light" style={{color:'var(--blue)'}}>{m.tmdbRating.toFixed(1)}</span>
                </div>
                <p className="font-display text-[1rem] font-light leading-tight mb-1" style={{color:'var(--blue)'}}>{m.name}</p>
                <p className="font-body text-[0.7rem] text-[var(--sub)] mb-2">
                  {m.releaseYear} · {m.language} · {m.runtime}
                  {m.timesWatched>=2&&<span className="font-semibold ml-2" style={{color:'var(--gold)'}}>{m.timesWatched}× watched</span>}
                </p>
                <p className="font-body text-[0.75rem] text-[var(--sub)] leading-relaxed line-clamp-2">{m.overview}</p>
              </button>
            ))}
          </div>
        </div>

        {/* BROWSE */}
        <div>
          <div className="flex items-end justify-between mb-6">
            <div>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-2">Browse</p>
              <p className="font-display text-[1.5rem] sm:text-[2rem] font-light text-[var(--text)]">Collection</p>
            </div>
            <p className="font-body text-[0.75rem] text-[var(--muted)]">{filtered.length} films</p>
          </div>

          <div className="mb-6 space-y-4" style={{position:'relative',zIndex:20}}>
            <div className="flex gap-2 sm:gap-3 flex-wrap items-center">
              <div className="relative flex-1 min-w-[160px]">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="var(--muted)" strokeWidth="2">
                  <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                </svg>
                <input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search films or directors…"
                  className="w-full pl-9 pr-4 py-2.5 rounded-xl font-body text-[0.85rem] outline-none multiselect-input"
                  style={{background:'var(--surface)',border:'1px solid var(--fill-border)',color:'var(--text)'}}/>
              </div>
              <div className="flex gap-1.5 sm:gap-2 flex-wrap">
                {(['rating','rewatched','date'] as SortKey[]).map(k=>(
                  <button key={k} onClick={()=>handleSort(k)} style={btnStyle(sortKey===k)}>
                    {k==='rating'?'Rating':k==='rewatched'?'Rewatched':'Date'}{sortArrow(k)}
                  </button>
                ))}
              </div>
              <button onClick={()=>setShowRewatched(!showRewatched)} style={{...btnStyle(showRewatched), display:'flex', alignItems:'center', gap:'5px'}}>
                <svg width="12" height="11" viewBox="0 0 24 24" fill={showRewatched ? 'white' : 'none'} stroke={showRewatched ? 'white' : 'var(--sub)'} strokeWidth="2">
                  <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                </svg>
                Favourites
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="glass rounded-xl p-4" style={{overflow:'visible', position:'relative', zIndex:30}}>
                <MultiSelect label="Genre" options={allGenreOpts} selected={genres} onChange={setGenres} placeholder="Search genres…"/>
              </div>
              <div className="glass rounded-xl p-4" style={{overflow:'visible', position:'relative', zIndex:20}}>
                <MultiSelect label="Language" options={allLangOpts} selected={languages} onChange={setLanguages} placeholder="Search languages…"/>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2" style={{position:'relative',zIndex:10}}>
            {filtered.slice(0,60).map(m=>{
              const accentColor = m.tmdbRating>=8.5?'#34c759':m.tmdbRating>=7.5?'#0071e3':m.tmdbRating>=6.5?'#ff9500':'#ff3b30'
              return (
              <button key={m.name} onClick={()=>setSelectedMovie(m)}
                className="glass rounded-xl flex items-center gap-3 sm:gap-5 hover:bg-white/90 transition-all text-left w-full"
                style={{paddingTop:'14px',paddingBottom:'14px',paddingRight:'20px',paddingLeft:0,overflow:'hidden',position:'relative'}}>
                <div style={{position:'absolute',left:0,top:0,bottom:0,width:'3px',background:accentColor}} />
                <div className="font-display text-[1.5rem] font-light w-10 sm:w-12 text-center flex-shrink-0" style={{color:accentColor,marginLeft:'20px'}}>
                  {m.tmdbRating>0?m.tmdbRating.toFixed(1):'—'}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-body text-[0.85rem] font-medium text-[var(--text)] truncate">
                    {m.name}{m.timesWatched>=2&&<span style={{color:'var(--gold)'}} className="ml-1">★</span>}
                  </p>
                  <p className="font-body text-[0.7rem] text-[var(--sub)]">{m.releaseYear} · {m.director.split(',')[0].trim()} · {m.runtime}</p>
                </div>
                <div className="flex gap-2 flex-shrink-0 items-center">
                  {m.timesWatched>=2&&<span className="font-body text-[0.6rem] px-2 py-1 rounded-full font-semibold" style={{background:'var(--gold-bg)',color:'var(--gold)'}}>{m.timesWatched}×</span>}
                  <span className="hidden sm:inline font-body text-[0.6rem] px-2 py-1 rounded-full" style={{background:'var(--fill)',color:'var(--sub)'}}>{m.language}</span>
                  <span className="hidden sm:inline font-body text-[0.6rem] px-2 py-1 rounded-full" style={{background:'var(--fill)',color:'var(--sub)'}}>{m.genre.split(',')[0].trim()}</span>
                </div>
              </button>
            )})}

            {filtered.length > 60 && (
              <div className="py-6 text-center">
                <p className="font-body text-[0.75rem] text-[var(--muted)] mb-2">Showing 60 of {filtered.length} films</p>
                <Link href="/stats#section-catalogue" className="font-body text-[0.75rem] font-medium text-[var(--blue)] hover:opacity-70 transition-opacity">
                  View full collection →
                </Link>
              </div>
            )}
          </div>
        </div>

        <div className="mt-12 sm:mt-16 pt-6 text-center" style={{borderTop:'1px solid var(--separator)'}}>
          <p className="font-body text-[0.6rem] tracking-[0.12em] uppercase text-[var(--muted)]">
            {stats.total as number} films · {new Date().toLocaleDateString('en-US',{month:'long',year:'numeric'})}
          </p>
        </div>
      </div>

      {/* FLOATING CHAT */}
      {!chatOpen && (
        <button onClick={()=>setChatOpen(true)}
          className="fixed bottom-8 right-8 z-50 w-12 h-12 rounded-full flex items-center justify-center text-white transition-all hover:opacity-90"
          style={{background:'var(--blue)',boxShadow:'0 4px 16px rgba(0,0,0,0.2)'}}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
          </svg>
        </button>
      )}

      <ScrollJump />
      {selectedMovie && <MovieModal movie={selectedMovie} onClose={()=>setSelectedMovie(null)}/>}
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
      {chatOpen && <ChatPanel movies={allMovies} initialMessage={initialMsg} onClose={()=>{setChatOpen(false);setInitialMsg('')}}/>}
    </div>
  )
}
