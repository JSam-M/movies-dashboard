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

  // Daily page views (last 90 days)
  const daily: Record<string, number> = {}
  v.forEach(r => {
    const day = r.created_at.slice(0, 10)
    daily[day] = (daily[day] || 0) + 1
  })
  const dailyViews = Object.entries(daily)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-90)
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

  // Week-over-week trend
  const now = new Date()
  const last7Start = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
  const prev7Start = new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000)
  const last7 = v.filter(r => new Date(r.created_at) >= last7Start).length
  const prev7 = v.filter(r => {
    const d = new Date(r.created_at)
    return d >= prev7Start && d < last7Start
  }).length

  // Hourly distribution (UTC)
  const hourly: Record<number, number> = {}
  v.forEach(r => {
    const h = new Date(r.created_at).getUTCHours()
    hourly[h] = (hourly[h] || 0) + 1
  })
  const hourlyViews = Array.from({ length: 24 }, (_, i) => ({ hour: i, count: hourly[i] || 0 }))

  // Day-of-week distribution (UTC)
  const DOW_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const dow: Record<number, number> = {}
  v.forEach(r => {
    const d = new Date(r.created_at).getUTCDay()
    dow[d] = (dow[d] || 0) + 1
  })
  const dowViews = DOW_LABELS.map((label, i) => ({ day: i, label, count: dow[i] || 0 }))

  // Peak day
  const sortedByCount = Object.entries(daily).sort((a, b) => b[1] - a[1])
  const peakDay = sortedByCount.length > 0
    ? { date: sortedByCount[0][0], count: sortedByCount[0][1] }
    : { date: '—', count: 0 }

  // Average daily views
  const uniqueDayCount = Object.keys(daily).length
  const avgDailyViews = uniqueDayCount > 0 ? Math.round(v.length / uniqueDayCount) : 0

  return NextResponse.json({
    totalViews: v.length,
    uniqueVisitors,
    returningVisitors,
    chatOpens,
    chatQueries,
    dailyViews,
    topPages,
    last7,
    prev7,
    hourlyViews,
    dowViews,
    peakDay,
    avgDailyViews,
  })
}
