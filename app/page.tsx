'use client'

import { useState, useEffect, useCallback } from 'react'
import type { Movie } from '@/lib/movies'
import KPIBar from '@/components/KPIBar'
import FilterBar from '@/components/FilterBar'
import CatalogueTab from '@/components/CatalogueTab'
import RankingsTab from '@/components/RankingsTab'
import CompositionTab from '@/components/CompositionTab'
import TrendsTab from '@/components/TrendsTab'
import ChatPanel from '@/components/ChatPanel'

export default function Home() {
  const [allMovies,   setAllMovies]   = useState<Movie[]>([])
  const [allEntries,  setAllEntries]  = useState<Movie[]>([])
  const [filtered,    setFiltered]    = useState<Movie[]>([])
  const [stats,       setStats]       = useState<Record<string, unknown>>({})
  const [activeTab,   setActiveTab]   = useState('catalogue')
  const [chatOpen,    setChatOpen]    = useState(false)
  const [loading,     setLoading]     = useState(true)

  // Filters
  const [search,    setSearch]    = useState('')
  const [language,  setLanguage]  = useState('All')
  const [genre,     setGenre]     = useState('All')
  const [director,  setDirector]  = useState('All')
  const [minRating, setMinRating] = useState(0)
  const [watchYear, setWatchYear] = useState<[number,number] | null>(null)
  const [rewatchFilter, setRewatchFilter] = useState('All')

  useEffect(() => {
    fetch('/api/movies')
      .then(r => r.json())
      .then(({ movies, allEntries: ae, stats: s }) => {
        setAllMovies(movies)
        setAllEntries(ae)
        setStats(s)
        setFiltered(movies)
        setLoading(false)
      })
  }, [])

  const applyFilters = useCallback(() => {
    let f = [...allMovies]
    if (search)      f = f.filter(m => m.name.toLowerCase().includes(search.toLowerCase()))
    if (language !== 'All') f = f.filter(m => m.language === language)
    if (genre !== 'All')    f = f.filter(m => m.genre.includes(genre))
    if (director !== 'All') f = f.filter(m => m.director.includes(director))
    if (minRating > 0)      f = f.filter(m => m.tmdbRating >= minRating)
    if (watchYear) {
      const names = new Set(
        allEntries
          .filter(e => {
            const y = parseInt('20' + e.date.split('/')[2])
            return y >= watchYear[0] && y <= watchYear[1]
          })
          .map(e => e.name)
      )
      f = f.filter(m => names.has(m.name))
    }
    if (rewatchFilter === 'Rewatched')   f = f.filter(m => m.timesWatched >= 2)
    if (rewatchFilter === 'First watch') f = f.filter(m => m.timesWatched <= 1)
    setFiltered(f)
  }, [allMovies, allEntries, search, language, genre, director, minRating, watchYear, rewatchFilter])

  useEffect(() => { applyFilters() }, [applyFilters])

  const tabs = ['catalogue','rankings','composition','trends']

  // Derived options
  const languages  = ['All', ...Array.from(new Set(allMovies.map(m => m.language))).sort()]
  const genres     = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.genre.split(',').map(g => g.trim()).filter(Boolean)))).sort()]
  const directors  = ['All', ...Array.from(new Set(allMovies.flatMap(m => m.director.split(',').map(d => d.trim()).filter(Boolean)))).sort()]
  const watchYears = Array.from(new Set(allEntries.map(e => parseInt('20' + e.date.split('/')[2])).filter(y => !isNaN(y)))).sort()
  const minWY = watchYears[0] || 2019
  const maxWY = watchYears[watchYears.length - 1] || 2026

  if (loading) return (
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="text-center">
        <div className="font-display text-4xl font-light text-gray-400 mb-4">Loading</div>
        <div className="flex gap-2 justify-center">
          {[0,1,2].map(i => (
            <div key={i} className="w-2 h-2 rounded-full bg-blue-400"
              style={{ animation: `pulse-dot 1.2s ease ${i*0.2}s infinite` }} />
          ))}
        </div>
      </div>
    </div>
  )

  return (
    <div className="min-h-screen mesh-bg relative">
      <div className="relative z-10 max-w-[1380px] mx-auto px-12 py-12">

        {/* ── HEADER ── */}
        <div className="flex items-start justify-between mb-10 pb-8 border-b border-black/7">
          <div>
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-4">
              Personal Archive · Since 2019
            </p>
            <h1 className="font-display text-[clamp(2.8rem,5vw,5rem)] font-light leading-[0.95] tracking-tight text-[var(--text)]">
              A life in{' '}
              <em style={{
                fontStyle: 'italic',
                background: 'linear-gradient(135deg,#0071e3,#34aadc)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}>cinema</em>
            </h1>
            <p className="font-body text-[0.88rem] text-[var(--sub)] mt-4 max-w-sm leading-relaxed">
              {stats.total as number} films across {stats.languages as number} languages —{' '}
              {stats.totalHours as number} hours of storytelling logged since the first watch.
            </p>
          </div>

          {/* Highest rated callout */}
          {filtered.length > 0 && (() => {
            const top = [...filtered].sort((a,b) => b.tmdbRating - a.tmdbRating)[0]
            return (
              <div className="glass rounded-[22px] p-7 ml-8 min-w-[220px] hidden lg:block"
                style={{ boxShadow: '0 8px 32px rgba(0,125,250,.1),0 2px 8px rgba(0,0,0,.06)' }}>
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.14em] uppercase text-[var(--sub)] mb-3">
                  Highest Rated
                </p>
                <p className="font-display text-[1.35rem] font-light text-[var(--text)] leading-tight mb-1">
                  {top.name}
                </p>
                <p className="font-body text-[0.7rem] text-[var(--sub)] mb-4">
                  {top.releaseYear} · {top.genre.split(',')[0]}
                </p>
                <p className="font-display text-[2.8rem] font-light leading-none" style={{ color: 'var(--blue)' }}>
                  {top.tmdbRating.toFixed(1)}
                  <span className="font-body text-base text-[var(--muted)]"> / 10</span>
                </p>
              </div>
            )
          })()}
        </div>

        {/* ── FILTERS ── */}
        <FilterBar
          search={search} setSearch={setSearch}
          language={language} setLanguage={setLanguage} languages={languages}
          genre={genre} setGenre={setGenre} genres={genres}
          director={director} setDirector={setDirector} directors={directors}
          minRating={minRating} setMinRating={setMinRating}
          minWY={minWY} maxWY={maxWY}
          watchYear={watchYear} setWatchYear={setWatchYear}
          rewatchFilter={rewatchFilter} setRewatchFilter={setRewatchFilter}
          total={allMovies.length} filtered={filtered.length}
        />

        {/* ── KPI BAR ── */}
        <KPIBar movies={filtered} allEntries={allEntries} watchYear={watchYear} />

        {/* ── TABS ── */}
        <div className="mt-8 mb-0 border-b border-black/7 flex gap-0">
          {tabs.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)}
              className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase px-7 py-4 border-b-[1.5px] transition-colors"
              style={{
                color: activeTab === tab ? 'var(--text)' : 'rgba(0,0,0,0.3)',
                borderBottomColor: activeTab === tab ? 'var(--text)' : 'transparent',
                marginBottom: '-1px',
              }}>
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

        {/* ── FOOTER ── */}
        <div className="mt-16 pt-6 border-t border-black/7 text-center">
          <p className="font-body text-[0.65rem] tracking-[0.1em] uppercase text-[rgba(0,0,0,0.2)]">
            {stats.total as number} films · v1.0 · {new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
          </p>
        </div>
      </div>

      {/* ── AI CHAT BUTTON ── */}
      <button
        onClick={() => setChatOpen(true)}
        className="fixed bottom-8 right-8 z-50 flex items-center gap-2 px-5 py-3.5 rounded-full font-body text-sm font-semibold text-white transition-all"
        style={{
          background: 'linear-gradient(135deg,#0071e3,#34aadc)',
          boxShadow: '0 4px 20px rgba(0,113,227,0.4)',
        }}>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
        Ask for a Recommendation
      </button>

      {/* ── AI CHAT PANEL ── */}
      {chatOpen && (
        <ChatPanel
          movies={allMovies}
          onClose={() => setChatOpen(false)}
        />
      )}
    </div>
  )
}
