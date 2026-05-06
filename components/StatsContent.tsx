'use client'

import type { Movie } from '@/lib/movies'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, LineChart, Line, CartesianGrid
} from 'recharts'

interface Props {
  movies: Movie[]
  allEntries: Movie[]
  watchYears: number[]
}

const QUAL = ['#0071e3','#ff9500','#34c759','#ff3b30','#5856d6','#ff2d55','#af52de','#00c7be','#32ade6','#ffcc00']

const tt = {
  contentStyle: {
    background:'rgba(255,255,255,0.97)', border:'1px solid rgba(0,0,0,0.08)',
    borderRadius:'12px', boxShadow:'0 4px 16px rgba(0,0,0,0.08)',
    fontFamily:'inherit', fontSize:'12px'
  },
  cursor: { fill:'rgba(0,0,0,0.03)' }
}

function Section({ eyebrow, title, children }: { eyebrow: string; title: string; children: React.ReactNode }) {
  return (
    <div className="mb-14">
      <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-1">{eyebrow}</p>
      <p className="font-display text-[1.8rem] font-light text-[var(--text)] mb-6">{title}</p>
      {children}
    </div>
  )
}

function KPICard({ value, unit, label, sub, dot }: { value: string; unit?: string; label: string; sub?: string; dot: string }) {
  const isText = value.length > 5
  return (
    <div className="glass rounded-2xl p-6 relative overflow-hidden" style={{minHeight:'110px',display:'flex',flexDirection:'column',justifyContent:'flex-end'}}>
      <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{background:dot,opacity:0.8}} />
      <div className={`font-light leading-none tracking-tight text-[var(--text)] ${isText ? 'font-body text-[1.1rem]' : 'font-display text-[2.4rem]'}`}>
        {value}{unit && <sup className="font-body text-[0.75rem] text-[var(--muted)] align-super ml-0.5">{unit}</sup>}
      </div>
      <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">{label}</div>
      {sub && <div className="font-body text-[0.6rem] text-[var(--muted)] mt-0.5">{sub}</div>}
    </div>
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
  entries = entries.map(e => ({ ...e, runtimeMins: parseRuntime(e.runtime) }))

  const totalMins  = entries.reduce((s, e) => s + (e as typeof e & {runtimeMins:number}).runtimeMins, 0)
  const totalHours = Math.round(totalMins / 60)
  const totalDays  = (totalMins / 1440).toFixed(1)
  const avgRating  = movies.filter(m => m.tmdbRating > 0).length > 0
    ? (movies.filter(m => m.tmdbRating > 0).reduce((s,m) => s + m.tmdbRating, 0) / movies.filter(m => m.tmdbRating > 0).length).toFixed(1)
    : '0'
  const rewatched  = movies.filter(m => m.timesWatched >= 2)
  const highRated  = movies.filter(m => m.tmdbRating >= 7.5).length
  const pctHighRated = movies.length > 0 ? Math.round(highRated / movies.length * 100) : 0

  // Peak year
  const yearCount: Record<number,number> = {}
  entries.forEach(e => {
    const y = parseInt('20' + e.date.split('/')[2])
    if (!isNaN(y)) yearCount[y] = (yearCount[y]||0)+1
  })
  const peakEntry  = Object.entries(yearCount).sort((a,b)=>+b[1]-+a[1])[0]
  const peakYear   = peakEntry ? `${peakEntry[0]} · ${peakEntry[1]} films` : '—'

  // Top director
  const dirCount: Record<string,number> = {}
  movies.forEach(m => m.director.split(',').forEach(d => {
    const t = d.trim(); if (t && t !== 'N/A') dirCount[t] = (dirCount[t]||0)+1
  }))
  const topDir = Object.entries(dirCount).sort((a,b)=>b[1]-a[1])[0]
  const topDirStr = topDir ? `${topDir[0].split(' ').pop()} · ${topDir[1]}` : '—'

  // Most recent film
  const recent = [...entries].sort((a,b) => b.date.localeCompare(a.date))[0]
  const recentStr = recent ? `${recent.name.slice(0,18)}${recent.name.length>18?'…':''}` : '—'

  // ── Viewing timeline by year-month ──
  const monthData: Record<string,{movies:number,hours:number}> = {}
  entries.forEach((e: typeof e & {runtimeMins?:number}) => {
    const [d,mo,y] = e.date.split('/')
    if (d&&mo&&y) {
      const k = `20${y}-${mo.padStart(2,'0')}`
      monthData[k] = monthData[k]||{movies:0,hours:0}
      monthData[k].movies++
      monthData[k].hours += ((e as typeof e & {runtimeMins:number}).runtimeMins||0)/60
    }
  })
  const timelineData = Object.entries(monthData).sort((a,b)=>a[0].localeCompare(b[0]))
    .map(([k,v]) => ({ period: k, films: v.movies, hours: Math.round(v.hours) }))

  // Annual summary
  const annualData = Object.entries(yearCount).sort((a,b)=>+a[0]-+b[0])
    .map(([y,c]) => ({ year: y, films: c }))

  // Language
  const langData = Object.entries(movies.reduce((a,m)=>{a[m.language]=(a[m.language]||0)+1;return a},{} as Record<string,number>))
    .sort((a,b)=>b[1]-a[1]).map(([name,value])=>({name,value}))

  // Genre
  const gc: Record<string,number> = {}
  movies.forEach(m => m.genre.split(',').forEach(g=>{const t=g.trim();if(t)gc[t]=(gc[t]||0)+1}))
  const genreData = Object.entries(gc).sort((a,b)=>b[1]-a[1]).slice(0,10)
    .map(([name,value])=>({name,value}))

  // Directors
  const dirData = Object.entries(dirCount).sort((a,b)=>b[1]-a[1]).slice(0,12)
    .map(([name,value])=>({name:name.length>24?name.slice(0,21)+'…':name,value}))

  // Rating distribution
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

  // Top rewatched
  const topRewatched = rewatched.sort((a,b)=>b.timesWatched-a.timesWatched).slice(0,8)
    .map(m=>({name:m.name.length>26?m.name.slice(0,23)+'…':m.name, times:m.timesWatched, rating:m.tmdbRating}))

  return (
    <div>
      {/* Page header */}
      <div className="mb-10 pb-8 border-b border-black/7">
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Personal Archive</p>
        <h1 className="font-display text-[2.8rem] font-light text-[var(--text)] leading-tight">My Film Stats</h1>
      </div>

      {/* ── KPIs ── */}
      <Section eyebrow="Overview" title="At a Glance">
        <div className="grid grid-cols-4 gap-4 mb-4">
          <KPICard value={String(movies.length)}   label="Unique Films"       dot="#0071e3" />
          <KPICard value={String(entries.length)}  label="Total Watches"      dot="#5856d6" sub="incl. rewatches" />
          <KPICard value={avgRating}               label="Avg TMDb Rating"    dot="#ff9500" sub="out of 10" />
          <KPICard value={pctHighRated+"%"}        label="Rated ≥7.5"         dot="#34c759" sub={`${highRated} films`} />
        </div>
        <div className="grid grid-cols-4 gap-4">
          <KPICard value={String(totalDays)} unit="d" label="Days in Cinema"  dot="#ff3b30" sub={`${totalHours}h total`} />
          <KPICard value={String(rewatched.length)} label="Rewatched"         dot="#af52de" sub="personal picks" />
          <KPICard value={topDirStr}               label="Top Director"       dot="#00c7be" />
          <KPICard value={peakYear}                label="Peak Year"          dot="#ffcc00" />
        </div>
      </Section>

      {/* ── TIMELINE ── */}
      <Section eyebrow="Trends" title="Viewing Over Time">
        <div className="glass rounded-2xl p-6 mb-4">
          <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Films per Month</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={timelineData} margin={{left:0,right:8,top:4,bottom:40}}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
              <XAxis dataKey="period" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} angle={-45} textAnchor="end" interval={5} />
              <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Line type="monotone" dataKey="films" stroke="#0071e3" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="glass rounded-2xl p-6">
          <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Films per Year</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={annualData} margin={{left:0,right:8,top:4,bottom:0}}>
              <XAxis dataKey="year" tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Bar dataKey="films" radius={[4,4,0,0]} fill="#0071e3"
                label={{position:'top',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ── LANGUAGE & GENRE ── */}
      <Section eyebrow="Composition" title="Language & Genre">
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6">
            <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">By Language</p>
            <div className="flex items-center gap-4">
              <PieChart width={160} height={160}>
                <Pie data={langData} cx={80} cy={80} innerRadius={50} outerRadius={75} dataKey="value" paddingAngle={2}>
                  {langData.map((_,i)=><Cell key={i} fill={QUAL[i%QUAL.length]} />)}
                </Pie>
                <Tooltip {...tt} />
              </PieChart>
              <div className="flex-1 space-y-1.5">
                {langData.slice(0,6).map((l,i)=>(
                  <div key={l.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full flex-shrink-0" style={{background:QUAL[i%QUAL.length]}} />
                      <span className="font-body text-[0.75rem] text-[var(--text)]">{l.name}</span>
                    </div>
                    <span className="font-body text-[0.72rem] text-[var(--sub)]">{l.value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div className="glass rounded-2xl p-6">
            <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Top Genres</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={[...genreData].reverse()} layout="vertical" margin={{left:0,right:36,top:0,bottom:0}}>
                <XAxis type="number" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={110} tick={{fontFamily:'inherit',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
                <Tooltip {...tt} />
                <Bar dataKey="value" radius={[0,4,4,0]} fill="#0071e3"
                  label={{position:'right',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      {/* ── RATING DISTRIBUTION ── */}
      <Section eyebrow="Quality" title="Rating Distribution">
        <div className="grid grid-cols-2 gap-6">
          <div className="glass rounded-2xl p-6">
            <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Films by TMDb Score</p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={ratingData} margin={{left:0,right:8,top:4,bottom:0}}>
                <XAxis dataKey="name" tick={{fontFamily:'inherit',fontSize:11,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <YAxis tick={{fontFamily:'inherit',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <Tooltip {...tt} />
                <Bar dataKey="value" radius={[4,4,0,0]} fill="#0071e3"
                  label={{position:'top',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top rated */}
          <div className="glass rounded-2xl p-6">
            <p className="font-body text-[0.65rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Highest Rated</p>
            <div className="space-y-2.5">
              {[...movies].sort((a,b)=>b.tmdbRating-a.tmdbRating).slice(0,6).map((m,i)=>(
                <div key={m.name} className="flex items-center gap-3">
                  <span className="font-body text-[0.65rem] text-[var(--muted)] w-4">{i+1}</span>
                  <div className="flex-1 min-w-0">
                    <p className="font-body text-[0.82rem] font-medium text-[var(--text)] truncate">{m.name}</p>
                    <p className="font-body text-[0.68rem] text-[var(--sub)]">{m.releaseYear} · {m.genre.split(',')[0]}</p>
                  </div>
                  <span className="font-display text-[1.2rem] font-light flex-shrink-0" style={{color:'var(--blue)'}}>{m.tmdbRating.toFixed(1)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ── DIRECTORS ── */}
      <Section eyebrow="Filmmakers" title="Top Directors">
        <div className="glass rounded-2xl p-6">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={[...dirData].reverse()} layout="vertical" margin={{left:0,right:36,top:0,bottom:0}}>
              <XAxis type="number" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={170} tick={{fontFamily:'inherit',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
              <Tooltip {...tt} />
              <Bar dataKey="value" radius={[0,4,4,0]} fill="#00c7be"
                label={{position:'right',fontSize:10,fill:'#86868b',fontFamily:'inherit'}} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      {/* ── REWATCHED ── */}
      {topRewatched.length > 0 && (
        <Section eyebrow="Personal Picks" title="Most Rewatched">
          <div className="glass rounded-2xl p-6">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={[...topRewatched].reverse()} layout="vertical" margin={{left:0,right:40,top:0,bottom:0}}>
                <XAxis type="number" tick={{fontFamily:'inherit',fontSize:9,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={180} tick={{fontFamily:'inherit',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
                <Tooltip {...tt} formatter={(v:number)=>[`${v}×`,'Times watched']} />
                <Bar dataKey="times" radius={[0,4,4,0]} fill="#ff9500"
                  label={{position:'right',fontSize:11,fill:'#86868b',fontFamily:'inherit',formatter:(v:number)=>`${v}×`}} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Section>
      )}

      {/* ── CATALOGUE ── */}
      <Section eyebrow="Browse" title="Complete Catalogue">
        <div className="glass rounded-2xl overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr style={{borderBottom:'1px solid rgba(0,0,0,0.06)'}}>
                {['Film','Year','Rating','Watches','Genre','Director','Runtime','Language'].map(h => (
                  <th key={h} className="px-4 py-3 text-left font-body text-[0.58rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)]">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...movies].sort((a,b)=>b.tmdbRating-a.tmdbRating).map((m,i) => (
                <tr key={m.name} style={{borderBottom: i<movies.length-1 ? '1px solid rgba(0,0,0,0.04)' : 'none'}}
                  className="hover:bg-black/[0.02] transition-colors">
                  <td className="px-4 py-2.5 font-body text-[0.82rem] font-medium text-[var(--text)]">
                    {m.name} {m.timesWatched>=2 && <span className="text-amber-400 ml-1">★</span>}
                  </td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.releaseYear}</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.tmdbRating.toFixed(1)}</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.timesWatched}×</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.genre.split(',').slice(0,2).join(', ')}</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.director.split(',')[0].trim()}</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.runtime}</td>
                  <td className="px-4 py-2.5 font-body text-[0.78rem] text-[var(--sub)]">{m.language}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <div className="pt-6 border-t border-black/7 text-center pb-8">
        <p className="font-body text-[0.65rem] tracking-[0.1em] uppercase text-[rgba(0,0,0,0.2)]">
          {movies.length} films · v2.0 · {new Date().toLocaleDateString('en-US',{month:'long',year:'numeric'})}
        </p>
      </div>
    </div>
  )
}
