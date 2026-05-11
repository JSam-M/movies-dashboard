'use client'

import type { Movie } from '@/lib/movies'
import React from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid
} from 'recharts'

interface Props {
  movies: Movie[]
  allEntries: Movie[]
  watchYears: number[]
}

const QUAL = ['#0071e3','#ff9500','#34c759','#ff3b30','#5856d6','#ff2d55','#af52de','#32ade6','#30b0c7','#ffcc00']

const tt = {
  contentStyle: {
    background:'var(--modal-bg)', border:'1px solid var(--fill-border)',
    borderRadius:'12px', boxShadow:'0 4px 16px rgba(0,0,0,0.08)',
    fontFamily:'inherit', fontSize:'12px'
  },
  labelStyle: { color:'var(--text)', fontWeight:500 },
  itemStyle: { color:'var(--text)' },
  cursor: { fill:'var(--fill)' }
}

function scrollTo(id: string) {
  const el = document.getElementById(id)
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function Section({ eyebrow, title, children, id }: { eyebrow: string; title: string; children: React.ReactNode; id?: string }) {
  return (
    <div className="mb-14" id={id}>
      <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">{eyebrow}</p>
      <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-6">{title}</p>
      {children}
    </div>
  )
}

function KPICard({ value, unit, label, sub, dot, sectionId }: {
  value: string; unit?: string; label: string; sub?: string; dot: string; sectionId?: string
}) {
  const isText = value.length > 5
  const clickable = !!sectionId
  return (
    <div
      className={`glass rounded-2xl p-6 relative overflow-hidden transition-all${clickable ? ' cursor-pointer hover:bg-white/90' : ''}`}
      style={{ height: '130px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}
      onClick={() => sectionId && scrollTo(sectionId)}
      title={clickable ? 'Click to view detail' : undefined}
    >
      <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{ background: dot, opacity: 0.8 }} />
      <div className={`font-light leading-none tracking-tight text-[var(--text)] ${isText ? 'font-body text-[1.1rem]' : 'font-display text-[2.4rem]'}`}>
        {value}{unit && <sup className="font-body text-[0.75rem] text-[var(--muted)] align-super ml-0.5">{unit}</sup>}
      </div>
      <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">{label}</div>
      {sub && <div className="font-body text-[0.6rem] text-[var(--muted)] mt-0.5">{sub}</div>}
    </div>
  )
}

function CatalogueSection({ movies }: { movies: Movie[] }) {
  const [sortCol, setSortCol] = React.useState<'name'|'releaseYear'|'tmdbRating'|'timesWatched'|'genre'|'director'|'runtime'|'language'>('tmdbRating')
  const [sortDir, setSortDir] = React.useState<'asc'|'desc'>('desc')
  const [selectedMovie, setSelectedMovie] = React.useState<Movie|null>(null)

  const handleSort = (col: typeof sortCol) => {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    else { setSortCol(col); setSortDir(col === 'name' || col === 'genre' || col === 'director' || col === 'language' ? 'asc' : 'desc') }
  }

  const sorted = [...movies].sort((a, b) => {
    const va = a[sortCol], vb = b[sortCol]
    const cmp = typeof va === 'string' ? String(va).localeCompare(String(vb)) : (Number(va) - Number(vb))
    return sortDir === 'asc' ? cmp : -cmp
  })

  const arrow = (col: typeof sortCol) => sortCol === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : ''

  const thStyle = (col: typeof sortCol): React.CSSProperties => ({
    padding:'10px 14px', textAlign:'left', fontSize:'0.6rem', fontWeight:600,
    letterSpacing:'0.12em', textTransform:'uppercase',
    color: sortCol === col ? '#0071e3' : '#86868b',
    cursor:'pointer', userSelect:'none', whiteSpace:'nowrap',
    borderBottom:'1px solid var(--separator)', fontFamily:'inherit',
  })

  return (
    <Section eyebrow="Browse" title="Complete Catalogue">
      {/* Modal */}
      {selectedMovie && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 sm:p-8"
          style={{background:'rgba(0,0,0,0.25)',backdropFilter:'blur(12px)'}}
          onClick={() => setSelectedMovie(null)}>
          <div className="relative w-full animate-fade-up rounded-3xl"
            style={{maxWidth:'480px',background:'var(--modal-bg)',padding:'28px',boxShadow:'0 32px 80px rgba(0,0,0,0.18)',border:'1px solid var(--glass-border)'}}
            onClick={e => e.stopPropagation()}>
            <button onClick={() => setSelectedMovie(null)} className="absolute top-5 right-5 text-[var(--muted)] hover:text-[var(--text)] transition-colors">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
              </svg>
            </button>
            <div className="flex items-start justify-between mb-4 pr-8">
              <div>
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase px-2 py-1 rounded-full mb-3 inline-block"
                  style={{background:'rgba(0,113,227,0.07)',color:'var(--blue)'}}>
                  {selectedMovie.genre.split(',')[0].trim()}
                </p>
                <h2 className="font-display text-[1.5rem] font-light text-[var(--text)] leading-tight">{selectedMovie.name}</h2>
                <p className="font-body text-[0.75rem] text-[var(--sub)] mt-1">
                  {selectedMovie.releaseYear} · {selectedMovie.language} · {selectedMovie.runtime}
                  {selectedMovie.timesWatched >= 2 && <span style={{color:'var(--gold)'}} className="ml-2">★ Watched {selectedMovie.timesWatched}×</span>}
                </p>
              </div>
              <div className="text-right flex-shrink-0 ml-4">
                <div className="font-display text-[2.2rem] font-light" style={{color:'var(--blue)'}}>{selectedMovie.tmdbRating.toFixed(1)}</div>
                <div className="font-body text-[0.6rem] text-[var(--muted)]">TMDb</div>
              </div>
            </div>
            <p className="font-body text-[0.85rem] text-[var(--sub)] leading-relaxed mb-5">{selectedMovie.overview || 'No overview available.'}</p>
            <div className="flex flex-wrap gap-2 pt-4" style={{borderTop:'1px solid var(--separator)'}}>
              <span className="font-body text-[0.7rem] text-[var(--sub)]">Director:</span>
              <span className="font-body text-[0.7rem] text-[var(--text)]">{selectedMovie.director.split(',')[0].trim()}</span>
              {selectedMovie.genre.split(',').slice(0,3).map(g => (
                <span key={g} className="font-body text-[0.6rem] px-2 py-0.5 rounded-full"
                  style={{background:'var(--fill)',color:'var(--sub)'}}>{g.trim()}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Desktop table */}
      <div className="glass rounded-2xl overflow-x-auto hidden sm:block">
        <table style={{width:'100%',borderCollapse:'collapse',fontSize:'0.85rem'}}>
          <thead>
            <tr>
              <th style={thStyle('name')} onClick={() => handleSort('name')}>Film{arrow('name')}</th>
              <th style={thStyle('releaseYear')} onClick={() => handleSort('releaseYear')}>Year{arrow('releaseYear')}</th>
              <th style={thStyle('tmdbRating')} onClick={() => handleSort('tmdbRating')}>Rating{arrow('tmdbRating')}</th>
              <th style={thStyle('timesWatched')} onClick={() => handleSort('timesWatched')}>Watches{arrow('timesWatched')}</th>
              <th style={thStyle('genre')} onClick={() => handleSort('genre')}>Genre{arrow('genre')}</th>
              <th style={thStyle('director')} onClick={() => handleSort('director')}>Director{arrow('director')}</th>
              <th style={thStyle('runtime')} onClick={() => handleSort('runtime')}>Runtime{arrow('runtime')}</th>
              <th style={thStyle('language')} onClick={() => handleSort('language')}>Language{arrow('language')}</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <tr key={m.name} style={{borderBottom: i < sorted.length-1 ? '1px solid var(--separator)' : 'none'}}
                className="hover:bg-black/[0.02] transition-colors cursor-pointer"
                onClick={() => setSelectedMovie(m)}>
                <td style={{padding:'9px 14px',fontWeight:500,color:'var(--text)',fontFamily:'inherit'}}>
                  {m.name}{m.timesWatched>=2 && <span style={{color:'var(--gold)',marginLeft:'4px'}}>★</span>}
                </td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.releaseYear}</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.tmdbRating.toFixed(1)}</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.timesWatched}×</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.genre.split(',').slice(0,2).join(', ')}</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.director.split(',')[0].trim()}</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.runtime}</td>
                <td style={{padding:'9px 14px',color:'var(--sub)',fontFamily:'inherit'}}>{m.language}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Mobile card list */}
      <div className="grid grid-cols-1 gap-2 sm:hidden">
        {sorted.map(m => (
          <button key={m.name} onClick={() => setSelectedMovie(m)}
            className="glass rounded-xl px-4 py-3.5 flex items-center gap-3 hover:bg-white/90 transition-all text-left w-full">
            <div className="font-display text-[1.5rem] font-light w-10 text-center flex-shrink-0" style={{color:'var(--blue)'}}>
              {m.tmdbRating>0?m.tmdbRating.toFixed(1):'—'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-body text-[0.85rem] font-medium text-[var(--text)] truncate">
                {m.name}{m.timesWatched>=2&&<span style={{color:'var(--gold)'}} className="ml-1">★</span>}
              </p>
              <p className="font-body text-[0.7rem] text-[var(--sub)]">{m.releaseYear} · {m.director.split(',')[0].trim()} · {m.runtime}</p>
            </div>
            {m.timesWatched>=2&&(
              <span className="font-body text-[0.6rem] px-2 py-1 rounded-full font-semibold flex-shrink-0"
                style={{background:'var(--gold-bg)',color:'var(--gold)'}}>{m.timesWatched}×</span>
            )}
          </button>
        ))}
      </div>
    </Section>
  )
}

export default function StatsContent({ movies, allEntries, watchYears }: Props) {
  const names = new Set(movies.map(m => m.name))
  let entries = allEntries.filter(e => names.has(e.name))
  if (watchYears.length > 0) {
    entries = entries.filter(e => {
      const y = parseInt('20' + e.date.split('/')[2])
      return watchYears.includes(y)
    })
  }

  const parseRuntime = (v: string) => { const n = parseInt(v); return isNaN(n) ? 0 : n }
  const entriesWithMins = entries.map(e => ({ ...e, runtimeMins: parseRuntime(e.runtime) }))

  const totalMins  = entriesWithMins.reduce((s, e) => s + e.runtimeMins, 0)
  const totalHours = Math.round(totalMins / 60)
  const totalDays  = (totalMins / 1440).toFixed(1)
  const avgRating  = movies.filter(m => m.tmdbRating > 0).length > 0
    ? (movies.filter(m => m.tmdbRating > 0).reduce((s,m) => s + m.tmdbRating, 0) / movies.filter(m => m.tmdbRating > 0).length).toFixed(1)
    : '0'
  const rewatched  = movies.filter(m => m.timesWatched >= 2)
  const highRated  = movies.filter(m => m.tmdbRating >= 7.5).length
  const pctHighRated = movies.length > 0 ? Math.round(highRated / movies.length * 100) : 0

  const yearCount: Record<number,number> = {}
  entries.forEach(e => {
    const y = parseInt('20' + e.date.split('/')[2])
    if (!isNaN(y)) yearCount[y] = (yearCount[y]||0)+1
  })
  const peakEntry = Object.entries(yearCount).sort((a,b)=>+b[1]-+a[1])[0]
  const peakYear  = peakEntry ? `${peakEntry[0]} · ${peakEntry[1]} films` : '—'

  const dirCount: Record<string,number> = {}
  movies.forEach(m => m.director.split(',').forEach(d => {
    const t = d.trim(); if (t && t !== 'N/A') dirCount[t] = (dirCount[t]||0)+1
  }))
  const topDir    = Object.entries(dirCount).sort((a,b)=>b[1]-a[1])[0]
  const topDirStr = topDir ? `${topDir[0].split(' ').pop()} · ${topDir[1]}` : '—'

  const monthData: Record<string,{movies:number,hours:number}> = {}
  entriesWithMins.forEach((e) => {
    const [d,mo,y] = e.date.split('/')
    if (d&&mo&&y) {
      const k = `20${y}-${mo.padStart(2,'0')}`
      monthData[k] = monthData[k]||{movies:0,hours:0}
      monthData[k].movies++
      monthData[k].hours += (e.runtimeMins||0)/60
    }
  })
  const timelineData = Object.entries(monthData).sort((a,b)=>a[0].localeCompare(b[0]))
    .map(([k,v]) => ({ period: k, films: v.movies, hours: Math.round(v.hours) }))

  const annualData = Object.entries(yearCount).sort((a,b)=>+a[0]-+b[0])
    .map(([y,c]) => ({ year: y, films: c }))

  const langData = Object.entries(movies.reduce((a,m)=>{a[m.language]=(a[m.language]||0)+1;return a},{} as Record<string,number>))
    .sort((a,b)=>b[1]-a[1]).map(([name,value])=>({name,value}))

  const gc: Record<string,number> = {}
  movies.forEach(m => m.genre.split(',').forEach(g=>{const t=g.trim();if(t)gc[t]=(gc[t]||0)+1}))
  const genreData = Object.entries(gc).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([name,value])=>({name,value}))

  const dirData = Object.entries(dirCount).sort((a,b)=>b[1]-a[1]).slice(0,12)
    .map(([name,value])=>({name:name.length>24?name.slice(0,21)+'\u2026':name,value}))

  const ratingBuckets: Record<string,number> = {'<5':0,'5–6':0,'6–7':0,'7–8':0,'8–9':0,'9–10':0}
  movies.filter(m=>m.tmdbRating>0).forEach(m => {
    if (m.tmdbRating < 5) ratingBuckets['<5']++
    else if (m.tmdbRating < 6) ratingBuckets['5–6']++
    else if (m.tmdbRating < 7) ratingBuckets['6–7']++
    else if (m.tmdbRating < 8) ratingBuckets['7–8']++
    else if (m.tmdbRating < 9) ratingBuckets['8–9']++
    else ratingBuckets['9–10']++
  })
  const ratingData = Object.entries(ratingBuckets).map(([name,value])=>({name,value}))

  const topRewatched = rewatched.sort((a,b)=>b.timesWatched-a.timesWatched).slice(0,8)
    .map(m=>({name:m.name.length>26?m.name.slice(0,23)+'\u2026':m.name, times:m.timesWatched, rating:m.tmdbRating}))

  return (
    <div>
      <div className="mb-10 pb-8" style={{borderBottom:'1px solid var(--separator)'}}>
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-2">Personal Archive</p>
        <h1 className="font-display text-[2.8rem] font-light text-[var(--text)] leading-tight">My Film Stats</h1>
      </div>

      <Section eyebrow="Overview" title="At a Glance" id="section-overview">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
          <KPICard value={String(movies.length)}      label="Unique Films"    sectionId="section-catalogue"  dot="#0071e3" />
          <KPICard value={String(entries.length)}     label="Total Watches"   sectionId="section-trends"     dot="#5856d6" sub="incl. rewatches" />
          <KPICard value={avgRating}                  label="Avg TMDb Rating" sectionId="section-quality"    dot="#ff9500" sub="out of 10" />
          <KPICard value={pctHighRated+'%'}           label="Rated ≥7.5"      sectionId="section-quality"    dot="#34c759" sub={`${highRated} films`} />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <KPICard value={String(totalDays)} unit="d" label="Days in Cinema"  sectionId="section-trends"     dot="#ff3b30" sub={`${totalHours}h total`} />
          <KPICard value={String(rewatched.length)}   label="Rewatched"       sectionId="section-rewatched"  dot="#af52de" sub="personal picks" />
          <KPICard value={topDirStr}                  label="Top Director"    sectionId="section-directors"  dot="#32ade6" />
          <KPICard value={peakYear}                   label="Peak Year"       sectionId="section-trends"     dot="var(--yellow)" />
        </div>
      </Section>

      <Section eyebrow="Trends" title="Viewing Over Time" id="section-trends">
        <div className="glass rounded-2xl p-4 sm:p-6 mb-4">
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">Films per Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={timelineData} margin={{left:0,right:8,top:4,bottom:40}}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" />
              <XAxis dataKey="period" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} angle={-45} textAnchor="end" interval={5} />
              <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Line type="monotone" dataKey="films" name="Films" stroke="#0071e3" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="glass rounded-2xl p-4 sm:p-6">
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">Films per Year</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={annualData} margin={{left:0,right:8,top:4,bottom:0}}>
              <XAxis dataKey="year" tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Bar dataKey="films" name="Films" radius={[4,4,0,0]} fill="#0071e3"
                label={{position:'top',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section eyebrow="Composition" title="Language & Genre" id="section-composition">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-stretch">
          <div className="glass rounded-2xl p-4 sm:p-6 flex flex-col justify-center">
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">By Language</p>
            <div className="flex items-center gap-4">
              <PieChart width={180} height={180}>
                <Pie data={langData} cx={90} cy={90} innerRadius={56} outerRadius={82} dataKey="value" paddingAngle={2}>
                  {langData.map((_,i)=><Cell key={i} fill={QUAL[i%QUAL.length]} />)}
                </Pie>
                <Tooltip {...tt} />
              </PieChart>
              <div className="flex-1 space-y-2">
                {langData.slice(0,6).map((l,i)=>(
                  <div key={l.name} className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 min-w-0">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{background:QUAL[i%QUAL.length]}} />
                      <span className="font-body text-[0.75rem] text-[var(--text)] truncate">{l.name}</span>
                    </div>
                    <span className="font-body text-[0.7rem] text-[var(--sub)] flex-shrink-0">{l.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="glass rounded-2xl p-4 sm:p-6 flex flex-col justify-center">
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">Top Genres</p>
            <div className="space-y-1.5">
              {(() => {
                const max = genreData[0]?.value || 1
                return genreData.map((item, i) => (
                  <div key={item.name} className="flex items-center gap-3">
                    <span className="font-body text-[0.75rem] text-[var(--text)] text-right flex-shrink-0"
                      style={{width:'100px'}}>{item.name}</span>
                    <div className="flex-1 h-[18px] rounded-[4px] overflow-hidden" style={{background:'var(--fill)'}}>
                      <div className="h-full rounded-[4px] transition-all"
                        style={{width:`${(item.value/max)*100}%`, background:`hsl(${210+i*8},${75-i*3}%,${50+i*2}%)`}} />
                    </div>
                    <span className="font-body text-[0.7rem] text-[var(--muted)] flex-shrink-0"
                      style={{width:'24px'}}>{item.value}</span>
                  </div>
                ))
              })()}
            </div>
          </div>
        </div>
      </Section>

      <Section eyebrow="Quality" title="Rating Distribution" id="section-quality">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 items-stretch">
          <div className="glass rounded-2xl p-4 sm:p-6 flex flex-col">
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">Films by TMDb Score</p>
            <div className="flex-1" style={{minHeight:'200px'}}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={ratingData} margin={{left:0,right:8,top:4,bottom:0}}>
                  <XAxis dataKey="name" tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
                  <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
                  <Tooltip {...tt} />
                  <Bar dataKey="value" name="Films" radius={[4,4,0,0]} fill="#0071e3"
                    label={{position:'top',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="glass rounded-2xl p-4 sm:p-6">
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-4">Highest Rated</p>
            <div className="space-y-1.5">
              {[...movies].sort((a,b)=>b.tmdbRating-a.tmdbRating).slice(0,5).map((m,i)=>(
                <div key={m.name} className="flex items-center gap-3">
                  <span className="font-body text-[0.6rem] text-[var(--muted)] w-4">{i+1}</span>
                  <div className="flex-1 min-w-0">
                    <p className="font-body text-[0.85rem] font-medium text-[var(--text)] truncate">{m.name}</p>
                    <p className="font-body text-[0.7rem] text-[var(--sub)]">{m.releaseYear} · {m.genre.split(',')[0]}</p>
                  </div>
                  <span className="font-display text-[1.2rem] font-light flex-shrink-0" style={{color:'var(--blue)'}}>{m.tmdbRating.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      <Section eyebrow="Filmmakers" title="Top Directors" id="section-directors">
        <div className="glass rounded-2xl p-4 sm:p-6">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={[...dirData]} layout="vertical" margin={{left:0,right:36,top:0,bottom:0}}>
              <XAxis type="number" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={170} tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Bar dataKey="value" name="Films" radius={[0,4,4,0]} fill="#32ade6"
                label={{position:'right',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {topRewatched.length > 0 && (
        <Section eyebrow="Personal Picks" title="Most Rewatched" id="section-rewatched">
          <div className="glass rounded-2xl p-4 sm:p-6">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={[...topRewatched]} layout="vertical" margin={{left:0,right:40,top:0,bottom:0}}>
                <XAxis type="number" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={180} tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <Tooltip {...tt} formatter={(v:number)=>[`${v}×`,'Times watched']} />
                <Bar dataKey="times" radius={[0,4,4,0]} fill="#ff9500"
                  label={{position:'right',fontSize:11,fill:'#86868b',fontFamily:'inherit',formatter:(v:number)=>`${v}×`}} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      <div id="section-catalogue"><CatalogueSection movies={movies} /></div>

      <div className="pt-6 text-center pb-8" style={{borderTop:'1px solid var(--separator)'}}>
        <p className="font-body text-[0.6rem] tracking-[0.12em] uppercase text-[var(--muted)]">
          {movies.length} films · {new Date().toLocaleDateString('en-US',{month:'long',year:'numeric'})}
        </p>
      </div>
    </div>
  )
}
