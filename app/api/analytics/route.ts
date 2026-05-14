import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

export async function GET(req: NextRequest) {
  const password = req.headers.get('x-analytics-password')
  if (password !== process.env.ANALYTICS_PASSWORD) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  const [{ data: views }, { data: events }] = await Promise.all([
    supabase.from('page_views').select('ip_hash, created_at, path').order('created_at', { ascending: true }),
    supabase.from('chat_events').select('event_type, created_at').order('created_at', { ascending: true }),
  ])

  const v = views ?? []
  const e = events ?? []

  // Unique & returning visitors
  const ipCounts: Record<string, number> = {}
  v.forEach(r => { ipCounts[r.ip_hash] = (ipCounts[r.ip_hash] || 0) + 1 })
  const uniqueVisitors   = Object.keys(ipCounts).length
  const returningVisitors = Object.values(ipCounts).filter(c => c > 1).length

  // Daily page views (last 30 days)
  const daily: Record<string, number> = {}
  v.forEach(r => {
    const day = r.created_at.slice(0, 10)
    daily[day] = (daily[day] || 0) + 1
  })
  const dailyViews = Object.entries(daily)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-30)
    .map(([date, count]) => ({ date, count }))

  // Chat stats
  const chatOpens   = e.filter(r => r.event_type === 'chat_open').length
  const chatQueries = e.filter(r => r.event_type === 'chat_query').length

  // Top pages
  const pathCounts: Record<string, number> = {}
  v.forEach(r => { pathCounts[r.path] = (pathCounts[r.path] || 0) + 1 })
  const topPages = Object.entries(pathCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([path, count]) => ({ path, count }))

  return NextResponse.json({
    totalViews: v.length,
    uniqueVisitors,
    returningVisitors,
    chatOpens,
    chatQueries,
    dailyViews,
    topPages,
  })
}
