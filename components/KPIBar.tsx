'use client'

import type { Movie } from '@/lib/movies'

interface Props {
  movies: Movie[]
  allEntries: Movie[]
  watchYear: [number,number] | null
}

export default function KPIBar({ movies, allEntries, watchYear }: Props) {
  const names = new Set(movies.map(m => m.name))
  let entries = allEntries.filter(e => names.has(e.name))
  if (watchYear) {
    entries = entries.filter(e => {
      const y = parseInt('20' + e.date.split('/')[2])
      return y >= watchYear[0] && y <= watchYear[1]
    })
  }

  const totalHours = Math.round(entries.reduce((s, e) => s + e.runtimeMins, 0) / 60)
  const avgRating  = movies.filter(m => m.tmdbRating > 0).length > 0
    ? +(movies.filter(m => m.tmdbRating > 0).reduce((s,m) => s + m.tmdbRating, 0) / movies.filter(m => m.tmdbRating > 0).length).toFixed(1)
    : 0
  const rewatched  = movies.filter(m => m.timesWatched >= 2).length
  const langs      = new Set(movies.map(m => m.language)).size

  const kpis = [
    { val: movies.length, unit: '',  label: 'Films',        dot: '#0071e3' },
    { val: avgRating,     unit: '',  label: 'Avg Rating',   dot: '#ff9500' },
    { val: totalHours,    unit: 'h', label: 'Hours Watched',dot: '#34c759' },
    { val: rewatched,     unit: '',  label: 'Rewatched',    dot: '#ff3b30' },
    { val: langs,         unit: '',  label: 'Languages',    dot: '#5856d6' },
  ]

  return (
    <div className="grid grid-cols-5 gap-4 mt-8">
      {kpis.map(k => (
        <div key={k.label} className="glass rounded-2xl p-6 relative overflow-hidden"
          style={{ minHeight: '110px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
          <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{ background: k.dot, opacity: 0.7 }} />
          <div className="font-display text-[2.6rem] font-light leading-none tracking-tight text-[var(--text)]">
            {k.val}<sup className="font-body text-[0.85rem] text-[var(--muted)] align-super">{k.unit}</sup>
          </div>
          <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">
            {k.label}
          </div>
        </div>
      ))}
    </div>
  )
}
