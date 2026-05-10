'use client'

import type { Movie } from '@/lib/movies'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

export default function RankingsTab({ movies }: { movies: Movie[] }) {
  const top10 = [...movies].filter(m => m.tmdbRating > 0)
    .sort((a,b) => b.tmdbRating - a.tmdbRating).slice(0,10)
    .map(m => ({ name: m.name.length > 28 ? m.name.slice(0,25)+'…' : m.name, rating: m.tmdbRating, year: m.releaseYear }))

  const top10rw = [...movies].filter(m => m.timesWatched > 1)
    .sort((a,b) => b.timesWatched - a.timesWatched).slice(0,10)
    .map(m => ({ name: m.name.length > 28 ? m.name.slice(0,25)+'…' : m.name, times: m.timesWatched }))

  const tooltip = {
    contentStyle: { background:'var(--modal-bg)', border:'1px solid var(--fill-border)', borderRadius:'12px', boxShadow:'0 4px 16px rgba(0,0,0,0.1)', fontFamily:'DM Sans,sans-serif', fontSize:'12px', color:'var(--text)' },
    cursor: { fill: 'var(--fill)' },
  }

  return (
    <div className="grid grid-cols-2 gap-8">
      <div>
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">By Rating</p>
        <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-5">Highest Rated</p>
        <div className="glass rounded-2xl p-5">
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={[...top10].reverse()} layout="vertical" margin={{left:0,right:40,top:8,bottom:0}}>
              <XAxis type="number" domain={[0,10]} tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={160} tick={{fontFamily:'DM Sans',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
              <Tooltip {...tooltip} formatter={(v:number) => [v.toFixed(1),'Rating']} />
              <Bar dataKey="rating" radius={[0,4,4,0]} label={{ position:'right', formatter:(v:number)=>v.toFixed(1), fontSize:10, fill:'#86868b', fontFamily:'DM Sans' }}>
                {top10.map((_,i) => <Cell key={i} fill={`hsl(${210 + i*3},${80-i*3}%,${45+i*3}%)`} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div>
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Personal Picks</p>
        <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-5">Most Rewatched</p>
        <div className="glass rounded-2xl p-5">
          {top10rw.length > 0 ? (
            <ResponsiveContainer width="100%" height={360}>
              <BarChart data={[...top10rw].reverse()} layout="vertical" margin={{left:0,right:40,top:8,bottom:0}}>
                <XAxis type="number" tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="name" width={160} tick={{fontFamily:'DM Sans',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
                <Tooltip {...tooltip} formatter={(v:number) => [`${v}×`,'Watched']} />
                <Bar dataKey="times" radius={[0,4,4,0]} label={{ position:'right', formatter:(v:number)=>`${v}×`, fontSize:10, fill:'#86868b', fontFamily:'DM Sans' }}>
                  {top10rw.map((_,i) => <Cell key={i} fill={`hsl(${35 + i*2},${90-i*4}%,${50+i*2}%)`} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="h-[360px] flex items-center justify-center text-[var(--sub)] font-body text-[0.85rem]">
              No rewatched films in current selection
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
