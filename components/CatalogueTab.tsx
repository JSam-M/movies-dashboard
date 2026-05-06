'use client'

import { useState } from 'react'
import type { Movie } from '@/lib/movies'

export default function CatalogueTab({ movies }: { movies: Movie[] }) {
  const [sortBy, setSortBy] = useState<'tmdbRating'|'name'|'releaseYear'>('tmdbRating')
  const [asc, setAsc] = useState(false)

  const sorted = [...movies].sort((a, b) => {
    const va = a[sortBy], vb = b[sortBy]
    if (typeof va === 'string' && typeof vb === 'string')
      return asc ? va.localeCompare(vb) : vb.localeCompare(va)
    return asc ? (va as number) - (vb as number) : (vb as number) - (va as number)
  })

  const cols: { key: 'tmdbRating'|'name'|'releaseYear', label: string }[] = [
    { key: 'name', label: 'Film' },
    { key: 'releaseYear', label: 'Year' },
    { key: 'tmdbRating', label: 'Rating' },
  ]

  return (
    <div>
      <div className="mb-5">
        <p className="font-body text-[0.6rem] font-semibold tracking-[0.14em] uppercase text-[var(--sub)] mb-1">Browse</p>
        <p className="font-display text-[1.6rem] font-light text-[var(--text)]">Complete Collection</p>
        <p className="font-body text-[0.75rem] text-[var(--sub)] mt-1">★ marks a personal rewatch</p>
      </div>

      <div className="glass rounded-2xl overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(0,0,0,0.06)' }}>
              {cols.map(c => (
                <th key={c.key}
                  onClick={() => { if(sortBy===c.key) setAsc(!asc); else { setSortBy(c.key); setAsc(false) } }}
                  className="px-5 py-3.5 text-left font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] cursor-pointer select-none hover:text-[var(--text)] transition-colors">
                  {c.label} {sortBy===c.key ? (asc?'↑':'↓') : ''}
                </th>
              ))}
              <th className="px-5 py-3.5 text-left font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)]">Genre</th>
              <th className="px-5 py-3.5 text-left font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)]">Director</th>
              <th className="px-5 py-3.5 text-left font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)]">Runtime</th>
              <th className="px-5 py-3.5 text-left font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)]">Language</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((m, i) => (
              <tr key={m.name} style={{ borderBottom: i < sorted.length-1 ? '1px solid rgba(0,0,0,0.04)' : 'none' }}
                className="hover:bg-black/[0.02] transition-colors">
                <td className="px-5 py-3 font-body text-[0.85rem] text-[var(--text)] font-medium">
                  {m.name} {m.timesWatched >= 2 ? <span className="text-amber-500 ml-1">★</span> : ''}
                </td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.releaseYear}</td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.tmdbRating.toFixed(1)}</td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.genre}</td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.director.split(',')[0]}</td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.runtime}</td>
                <td className="px-5 py-3 font-body text-[0.82rem] text-[var(--sub)]">{m.language}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
