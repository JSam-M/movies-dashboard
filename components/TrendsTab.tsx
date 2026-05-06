'use client'

import { useState } from 'react'
import type { Movie } from '@/lib/movies'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

interface Props { movies: Movie[]; allEntries: Movie[]; watchYear: [number,number] | null }

export default function TrendsTab({ movies, allEntries, watchYear }: Props) {
  const [view, setView] = useState<'year'|'month'|'all'>('year')

  const names = new Set(movies.map(m => m.name))
  let entries = allEntries.filter(e => names.has(e.name))
  if (watchYear) {
    entries = entries.filter(e => {
      const y = parseInt('20' + e.date.split('/')[2])
      return y >= watchYear[0] && y <= watchYear[1]
    })
  }

  // Build chart data
  const buildData = () => {
    if (view === 'year') {
      const acc: Record<number,{movies:number,mins:number}> = {}
      entries.forEach(e => {
        const y = parseInt('20' + e.date.split('/')[2])
        if (!isNaN(y)) { acc[y] = acc[y]||{movies:0,mins:0}; acc[y].movies++; acc[y].mins += e.runtimeMins }
      })
      return Object.entries(acc).sort((a,b)=>+a[0]-+b[0])
        .map(([k,v]) => ({ period: k, movies: v.movies, hours: +(v.mins/60).toFixed(0) }))
    }
    if (view === 'month') {
      const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
      const acc: Record<number,{movies:number,mins:number}> = {}
      entries.forEach(e => {
        const mo = parseInt(e.date.split('/')[1]) - 1
        if (!isNaN(mo)) { acc[mo] = acc[mo]||{movies:0,mins:0}; acc[mo].movies++; acc[mo].mins += e.runtimeMins }
      })
      return MONTHS.map((m,i) => ({ period: m, movies: acc[i]?.movies||0, hours: +(( acc[i]?.mins||0)/60).toFixed(0) }))
    }
    // all time — by year-month
    const acc: Record<string,{movies:number,mins:number}> = {}
    entries.forEach(e => {
      const [d,mo,y] = e.date.split('/')
      if (d&&mo&&y) { const k=`20${y}-${mo.padStart(2,'0')}`; acc[k]=acc[k]||{movies:0,mins:0}; acc[k].movies++; acc[k].mins+=e.runtimeMins }
    })
    return Object.entries(acc).sort((a,b)=>a[0].localeCompare(b[0]))
      .map(([k,v]) => ({ period: k, movies: v.movies, hours: +(v.mins/60).toFixed(0) }))
  }

  const data = buildData()
  const totWatches = entries.length
  const totHours   = Math.round(entries.reduce((s,e)=>s+e.runtimeMins,0)/60)
  const totDays    = +(totHours/24).toFixed(1)
  const avgRuntime = entries.length ? Math.round(entries.reduce((s,e)=>s+e.runtimeMins,0)/entries.length) : 0

  // Binge streak — parse DD/MM/YY correctly
  let binge = 0, bingeStr = ''
  const dates = [...new Set(entries.map(e => {
    const [d, m, y] = e.date.split('/')
    return `20${y}-${m.padStart(2,'0')}-${d.padStart(2,'0')}`
  }))].sort()
  if (dates.length > 0) {
    let streak = 1, best = 1, bestStart = dates[0], curStart = dates[0]
    for (let i = 1; i < dates.length; i++) {
      const a = new Date(dates[i-1]), b = new Date(dates[i])
      const diff = (b.getTime()-a.getTime())/(86400000)
      if (Math.round(diff) === 1) { streak++; if(streak>best){best=streak;bestStart=curStart} }
      else { streak=1; curStart=dates[i] }
    }
    binge = best
    const end = new Date(new Date(bestStart).getTime() + (best-1)*86400000)
    const fmt = (d:Date) => d.toLocaleDateString('en-GB',{day:'numeric',month:'short'})
    bingeStr = `${fmt(new Date(bestStart))} – ${fmt(end)}`
  }

  // Fav genre
  const gc: Record<string,number> = {}
  movies.forEach(m => m.genre.split(',').forEach(g=>{const t=g.trim();if(t)gc[t]=(gc[t]||0)+1}))
  const favGenre = Object.entries(gc).sort((a,b)=>b[1]-a[1])[0]?.[0] || '—'

  const trendKPIs = [
    { val: String(totWatches), unit: '', label: 'Total Watches', dot: '#0071e3' },
    { val: favGenre, unit: '', label: 'Fav Genre', dot: '#ff9500', small: true },
    { val: String(binge), unit: 'd', label: `Binge Streak${bingeStr ? '\n'+bingeStr : ''}`, dot: '#34c759' },
    { val: String(totDays), unit: 'd', label: 'Days in Cinema', dot: '#5856d6' },
    { val: String(avgRuntime), unit: 'm', label: 'Avg Runtime', dot: '#ff3b30' },
  ]

  const tooltip = {
    contentStyle: { background:'rgba(255,255,255,0.95)', border:'1px solid rgba(0,0,0,0.08)', borderRadius:'12px', boxShadow:'0 4px 16px rgba(0,0,0,0.1)', fontFamily:'DM Sans,sans-serif', fontSize:'12px' },
    cursor: { fill:'rgba(0,0,0,0.03)' },
  }

  return (
    <div>
      <p className="font-body text-[0.6rem] font-semibold tracking-[0.14em] uppercase text-[var(--sub)] mb-1">Analytics</p>
      <p className="font-display text-[1.6rem] font-light text-[var(--text)] mb-1">Viewing Over Time</p>
      <p className="font-body text-[0.75rem] text-[var(--sub)] mb-6">Films watched and hours invested across your collection history.</p>

      {/* View toggle */}
      <div className="flex gap-2 mb-6">
        {(['year','month','all'] as const).map(v => (
          <button key={v} onClick={()=>setView(v)}
            className="px-4 py-1.5 rounded-full font-body text-[0.75rem] font-medium transition-all"
            style={{
              background: view===v ? 'var(--blue)' : 'white',
              color: view===v ? 'white' : 'var(--sub)',
              border: `1px solid ${view===v ? 'var(--blue)' : 'rgba(0,0,0,0.1)'}`,
            }}>
            {v==='year'?'By Year':v==='month'?'By Month':'All Time'}
          </button>
        ))}
      </div>

      {/* Trend KPIs */}
      <div className="grid grid-cols-5 gap-4 mb-8">
        {trendKPIs.map(k => {
          const lines = k.label.split('\n')
          return (
            <div key={k.label} className="glass rounded-[18px] p-5 relative overflow-hidden"
              style={{ height:'110px', display:'flex', flexDirection:'column', justifyContent:'flex-end' }}>
              <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{background:k.dot,opacity:0.7}} />
              <div className={`font-display font-light leading-none tracking-tight text-[var(--text)] ${k.small ? 'text-[1.2rem]' : 'text-[2.2rem]'}`}
                style={k.small ? {fontFamily:'DM Sans,sans-serif'} : {}}>
                {k.val}<sup className="font-body text-[0.75rem] text-[var(--muted)] align-super">{k.unit}</sup>
              </div>
              <div className="font-body text-[0.58rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-1.5 leading-tight">
                {lines[0]}
                {lines[1] && <span className="block text-[0.52rem] normal-case tracking-normal font-normal opacity-70 mt-0.5">{lines[1]}</span>}
              </div>
            </div>
          )
        })}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-8">
        <div>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Volume</p>
          <div className="glass rounded-2xl p-5">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={data} margin={{left:0,right:8,top:8,bottom:40}}>
                <XAxis dataKey="period" tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} angle={view==='all'?-45:0} textAnchor={view==='all'?'end':'middle'} interval={view==='all'?5:0} />
                <YAxis tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <Tooltip {...tooltip} formatter={(v:number)=>[v,'Films']} />
                <Bar dataKey="movies" radius={[4,4,0,0]} fill="#0071e3"
                  label={view!=='all'?{position:'top',fontSize:10,fill:'#86868b',fontFamily:'DM Sans'}:false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mb-4">Duration</p>
          <div className="glass rounded-2xl p-5">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={data} margin={{left:0,right:8,top:8,bottom:40}}>
                <XAxis dataKey="period" tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} angle={view==='all'?-45:0} textAnchor={view==='all'?'end':'middle'} interval={view==='all'?5:0} />
                <YAxis tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <Tooltip {...tooltip} formatter={(v:number)=>[`${v}h`,'Hours']} />
                <Bar dataKey="hours" radius={[4,4,0,0]} fill="#ff9500"
                  label={view!=='all'?{position:'top',fontSize:10,fill:'#86868b',fontFamily:'DM Sans',formatter:(v:number)=>`${v}h`}:false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
