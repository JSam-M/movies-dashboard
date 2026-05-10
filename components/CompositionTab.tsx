'use client'

import type { Movie } from '@/lib/movies'
import { PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts'

const COLORS = ['#0071e3','#ff9500','#34c759','#ff3b30','#5856d6','#ff2d55','#af52de','#32ade6']

export default function CompositionTab({ movies }: { movies: Movie[] }) {
  const langData = Object.entries(
    movies.reduce((acc, m) => { acc[m.language] = (acc[m.language]||0)+1; return acc }, {} as Record<string,number>)
  ).sort((a,b)=>b[1]-a[1]).map(([name,value]) => ({name,value}))

  const genreData = Object.entries(
    movies.reduce((acc, m) => {
      m.genre.split(',').forEach(g => { const t=g.trim(); if(t) acc[t]=(acc[t]||0)+1 })
      return acc
    }, {} as Record<string,number>)
  ).sort((a,b)=>b[1]-a[1]).slice(0,10).map(([name,value]) => ({name,value}))

  const dirData = Object.entries(
    movies.reduce((acc, m) => {
      m.director.split(',').forEach(d => { const t=d.trim(); if(t&&t!=='N/A') acc[t]=(acc[t]||0)+1 })
      return acc
    }, {} as Record<string,number>)
  ).sort((a,b)=>b[1]-a[1]).slice(0,12).map(([name,value]) => ({name:name.length>22?name.slice(0,19)+'…':name,value}))

  const tooltip = {
    contentStyle: { background:'rgba(255,255,255,0.95)', border:'1px solid rgba(0,0,0,0.08)', borderRadius:'12px', boxShadow:'0 4px 16px rgba(0,0,0,0.1)', fontFamily:'DM Sans,sans-serif', fontSize:'12px' },
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-8">
        {/* Language donut */}
        <div>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Language</p>
          <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-5">By Language</p>
          <div className="glass rounded-2xl p-5 flex items-center justify-center">
            <PieChart width={320} height={300}>
              <Pie data={langData} cx={160} cy={140} innerRadius={70} outerRadius={110}
                dataKey="value" nameKey="name" paddingAngle={2}
                label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`}
                labelLine={false}>
                {langData.map((_,i) => <Cell key={i} fill={COLORS[i%COLORS.length]} />)}
              </Pie>
              <Tooltip {...tooltip} />
            </PieChart>
          </div>
        </div>

        {/* Genre bars — CSS implementation (bypasses Recharts axis rendering) */}
        <div>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Genre</p>
          <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-5">Top Genres</p>
          <div className="glass rounded-2xl p-5 space-y-2.5">
            {(() => {
              const max = genreData[0]?.value || 1
              return genreData.map((item, i) => (
                <div key={item.name} className="flex items-center gap-3">
                  <span className="font-body text-[0.75rem] text-[var(--text)] text-right flex-shrink-0"
                    style={{width:'90px'}}>{item.name}</span>
                  <div className="flex-1 h-[28px] rounded-[4px] overflow-hidden" style={{background:'rgba(0,0,0,0.04)'}}>
                    <div className="h-full rounded-[4px] transition-all"
                      style={{
                        width:`${(item.value/max)*100}%`,
                        background:`hsl(${210+i*8},${75-i*3}%,${50+i*2}%)`,
                      }} />
                  </div>
                  <span className="font-body text-[0.7rem] text-[var(--muted)] flex-shrink-0"
                    style={{width:'28px'}}>{item.value}</span>
                </div>
              ))
            })()}
          </div>
        </div>
      </div>

      {/* Directors */}
      <div>
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Filmmakers</p>
        <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-5">Top Directors</p>
        <div className="glass rounded-2xl p-5">
          <ResponsiveContainer width="100%" height={380}>
            <BarChart data={[...dirData].reverse()} layout="vertical" margin={{left:0,right:36,top:4,bottom:0}}>
              <XAxis type="number" tick={{fontFamily:'DM Sans',fontSize:10,fill:'#86868b'}} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={180} tick={{fontFamily:'DM Sans',fontSize:11,fill:'#1d1d1f'}} axisLine={false} tickLine={false} />
              <Tooltip {...tooltip} />
              <Bar dataKey="value" radius={[0,4,4,0]} label={{position:'right',fontSize:10,fill:'#86868b',fontFamily:'DM Sans'}}>
                {dirData.map((_,i) => <Cell key={i} fill={`hsl(${170+i*6},${65-i*2}%,${45+i*2}%)`} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
