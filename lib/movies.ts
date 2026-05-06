import fs from 'fs'
import path from 'path'
import Papa from 'papaparse'

export interface Movie {
  no: number
  date: string
  name: string
  language: string
  year: number
  good: string
  timesWatched: number
  location: string
  director: string
  runtime: string
  genre: string
  tmdbRating: number
  releaseYear: number
  overview: string
  apiStatus: string
  runtimeMins: number
}

function parseRuntime(val: string): number {
  if (!val) return 0
  const n = parseInt(val)
  return isNaN(n) ? 0 : n
}

let cachedMovies: Movie[] | null = null

export function getMovies(): Movie[] {
  if (cachedMovies) return cachedMovies

  const filePath = path.join(process.cwd(), 'public', 'movies.csv')
  const csv = fs.readFileSync(filePath, 'utf-8')

  const result = Papa.parse(csv, { header: true, skipEmptyLines: true })

  cachedMovies = (result.data as Record<string, string>[]).map(row => ({
    no:           parseInt(row['No.']) || 0,
    date:         row['Date'] || '',
    name:         row['Name'] || '',
    language:     row['Language'] || '',
    year:         parseInt(row['Year']) || 0,
    good:         row['Good?'] || '',
    timesWatched: parseFloat(row["N'th time of watching"]) || 1,
    location:     row['Location'] || '',
    director:     row['Director'] || '',
    runtime:      row['Runtime'] || '',
    genre:        row['Genre'] || '',
    tmdbRating:   parseFloat(row['TMDb_Rating']) || 0,
    releaseYear:  parseInt(row['Release_Year']) || 0,
    overview:     row['Overview'] || '',
    apiStatus:    row['API_Status'] || '',
    runtimeMins:  parseRuntime(row['Runtime']),
  }))

  return cachedMovies
}

export function getUniqueMovies(): Movie[] {
  const movies = getMovies()
  const seen = new Map<string, Movie>()
  // CSV is newest-first, so first occurrence = most recent watch
  for (const m of movies) {
    if (!seen.has(m.name)) seen.set(m.name, m)
    else {
      const existing = seen.get(m.name)!
      if (m.timesWatched > existing.timesWatched)
        seen.set(m.name, { ...existing, timesWatched: m.timesWatched })
    }
  }
  return Array.from(seen.values())
}

export function getStats(movies: Movie[], allEntries: Movie[]) {
  const names = new Set(movies.map(m => m.name))
  const entries = allEntries.filter(m => names.has(m.name))

  const totalHours = entries.reduce((s, m) => s + m.runtimeMins, 0) / 60
  const avgRating  = movies.filter(m => m.tmdbRating > 0)
                           .reduce((s, m, _, a) => s + m.tmdbRating / a.length, 0)

  const genreCount: Record<string, number> = {}
  movies.forEach(m => {
    m.genre.split(',').forEach(g => {
      const t = g.trim()
      if (t) genreCount[t] = (genreCount[t] || 0) + 1
    })
  })

  const langCount: Record<string, number> = {}
  movies.forEach(m => { langCount[m.language] = (langCount[m.language] || 0) + 1 })

  const yearCount: Record<number, number> = {}
  entries.forEach(m => {
    const y = parseInt(m.date.split('/')[2]) + 2000
    if (!isNaN(y)) yearCount[y] = (yearCount[y] || 0) + 1
  })

  const dirCount: Record<string, number> = {}
  movies.forEach(m => {
    m.director.split(',').forEach(d => {
      const t = d.trim()
      if (t && t !== 'N/A') dirCount[t] = (dirCount[t] || 0) + 1
    })
  })

  return {
    total: movies.length,
    totalEntries: entries.length,
    totalHours: Math.round(totalHours),
    totalDays: +(totalHours / 24).toFixed(1),
    avgRating: +avgRating.toFixed(1),
    rewatched: movies.filter(m => m.timesWatched >= 2).length,
    languages: Object.keys(langCount).length,
    genreCount,
    langCount,
    yearCount,
    dirCount,
  }
}
