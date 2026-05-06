import { NextResponse } from 'next/server'
import { getMovies, getUniqueMovies, getStats } from '@/lib/movies'

export const dynamic = 'force-static'
export const revalidate = 3600

export async function GET() {
  const all     = getMovies()
  const unique  = getUniqueMovies()
  const stats   = getStats(unique, all)
  return NextResponse.json({ movies: unique, allEntries: all, stats })
}
