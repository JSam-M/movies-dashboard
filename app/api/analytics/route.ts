import { NextRequest, NextResponse } from 'next/server'
import { timingSafeEqual } from 'crypto'
import { supabaseAdmin } from '@/lib/supabase'
import { rateLimit } from '@/lib/rateLimit'

function safeCompare(a: string, b: string): boolean {
  const ab = Buffer.from(a)
  const bb = Buffer.from(b)
  if (ab.length !== bb.length) return false
  return timingSafeEqual(ab, bb)
}

type ViewRow = {
  visitor_id: string
  created_at: string
  path: string
  device_type?: string | null
  country?: string | null
  referrer?: string | null
}

type EventRow = { event_type: string; created_at: string }

function getWeekMonday(dateStr: string): string {
  const d = new Date(dateStr)
  const day = d.getUTCDay()
  d.setUTCDate(d.getUTCDate() - (day === 0 ? 6 : day - 1))
  return d.toISOString().slice(0, 10)
}

export async function GET(req: NextRequest) {
  const ip = req.headers.get('x-forwarded-for')?.split(',')[0].trim() || 'unknown'
  const { allowed } = rateLimit(`analytics:${ip}`, { limit: 10, windowMs: 60_000 })
  if (!allowed) return NextResponse.json({ error: 'Too many requests' }, { status: 429 })

  const password = req.headers.get('x-analytics-password')
  const expected = process.env.ANALYTICS_PASSWORD
  if (!password || !expected || !safeCompare(password, expected)) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 })
  }

  // Try full select (post-migration), fall back to legacy columns
  let v: ViewRow[] = []
  const { data: fullViews, error: fullError } = await supabaseAdmin
    .from('page_views')
    .select('visitor_id, created_at, path, device_type, country, referrer')
    .order('created_at', { ascending: true })

  if (fullError) {
    const { data: legacyViews } = await supabaseAdmin
      .from('page_views')
      .select('ip_hash, created_at, path')
      .order('created_at', { ascending: true })
    v = (legacyViews ?? []).map((r: { ip_hash: string; created_at: string; path: string }) => ({
      visitor_id: r.ip_hash,
      created_at: r.created_at,
      path: r.path,
    }))
  } else {
    v = fullViews ?? []
  }

  const { data: events } = await supabaseAdmin
    .from('chat_events')
    .select('event_type, created_at')
    .order('created_at', { ascending: true })
  const e: EventRow[] = events ?? []

  // --- Visitor counts ---
  const visitorSeen: Record<string, number> = {}
  v.forEach(r => { visitorSeen[r.visitor_id] = (visitorSeen[r.visitor_id] || 0) + 1 })
  const uniqueVisitors = Object.keys(visitorSeen).length
  const returningVisitors = Object.values(visitorSeen).filter(c => c > 1).length

  // --- Daily views (last 90 days) ---
  const daily: Record<string, number> = {}
  v.forEach(r => {
    const day = r.created_at.slice(0, 10)
    daily[day] = (daily[day] || 0) + 1
  })
  const dailyViews = Object.entries(daily)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-90)
    .map(([date, count]) => ({ date, count }))

  // --- Monthly views (last 12 months) ---
  const monthly: Record<string, number> = {}
  v.forEach(r => {
    const month = r.created_at.slice(0, 7)
    monthly[month] = (monthly[month] || 0) + 1
  })
  const monthlyViews = Object.entries(monthly)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-12)
    .map(([month, count]) => ({ month, count }))

  // --- Chat stats ---
  const chatOpens = e.filter(r => r.event_type === 'chat_open').length
  const chatQueries = e.filter(r => r.event_type === 'chat_query').length

  // --- Top pages ---
  const pathCounts: Record<string, number> = {}
  v.forEach(r => { pathCounts[r.path] = (pathCounts[r.path] || 0) + 1 })
  const topPages = Object.entries(pathCounts)
    .sort((a, b) => b[1] - a[1])
    .map(([path, count]) => ({ path, count }))

  // --- Week / 30-day comparisons ---
  const now = new Date()
  const ms = (days: number) => days * 24 * 60 * 60 * 1000
  const last7Start  = new Date(now.getTime() - ms(7))
  const prev7Start  = new Date(now.getTime() - ms(14))
  const last30Start = new Date(now.getTime() - ms(30))
  const prev30Start = new Date(now.getTime() - ms(60))
  const last7  = v.filter(r => new Date(r.created_at) >= last7Start).length
  const prev7  = v.filter(r => { const d = new Date(r.created_at); return d >= prev7Start && d < last7Start }).length
  const last30 = v.filter(r => new Date(r.created_at) >= last30Start).length
  const prev30 = v.filter(r => { const d = new Date(r.created_at); return d >= prev30Start && d < last30Start }).length

  // --- Day × Hour heatmap (UTC) ---
  const dayHour: Record<string, number> = {}
  v.forEach(r => {
    const d = new Date(r.created_at)
    const key = `${d.getUTCDay()}_${d.getUTCHours()}`
    dayHour[key] = (dayHour[key] || 0) + 1
  })
  const dayHourHeatmap: { day: number; hour: number; count: number }[] = []
  for (let day = 0; day < 7; day++)
    for (let hour = 0; hour < 24; hour++)
      dayHourHeatmap.push({ day, hour, count: dayHour[`${day}_${hour}`] || 0 })

  // --- Peak day & averages ---
  const sortedByCount = Object.entries(daily).sort((a, b) => b[1] - a[1])
  const peakDay = sortedByCount.length > 0
    ? { date: sortedByCount[0][0], count: sortedByCount[0][1] }
    : { date: '—', count: 0 }
  const uniqueDayCount = Object.keys(daily).length
  const avgDailyViews = uniqueDayCount > 0 ? Math.round(v.length / uniqueDayCount) : 0

  // --- Device breakdown ---
  const devices: Record<string, number> = {}
  v.forEach(r => {
    const dt = r.device_type || 'unknown'
    devices[dt] = (devices[dt] || 0) + 1
  })
  const deviceBreakdown = Object.entries(devices)
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count)

  // --- Top countries ---
  const countries: Record<string, number> = {}
  v.forEach(r => { if (r.country) countries[r.country] = (countries[r.country] || 0) + 1 })
  const topCountries = Object.entries(countries)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([country, count]) => ({ country, count }))

  // --- Traffic sources ---
  const referrers: Record<string, number> = {}
  v.forEach(r => {
    const src = r.referrer || 'Direct'
    referrers[src] = (referrers[src] || 0) + 1
  })
  const topReferrers = Object.entries(referrers)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([source, count]) => ({ source, count }))

  // --- Session analysis (30-min window) ---
  const viewsByVisitor: Record<string, string[]> = {}
  v.forEach(r => {
    if (!viewsByVisitor[r.visitor_id]) viewsByVisitor[r.visitor_id] = []
    viewsByVisitor[r.visitor_id].push(r.created_at)
  })

  let totalSessions = 0, totalPages = 0, bounceSessions = 0
  let totalDurationMs = 0, durationCount = 0

  Object.values(viewsByVisitor).forEach(times => {
    times.sort()
    let sessionPages = 1
    let sessionStart = new Date(times[0]).getTime()
    let sessionLast = sessionStart

    for (let i = 1; i < times.length; i++) {
      const curr = new Date(times[i]).getTime()
      const gap = curr - new Date(times[i - 1]).getTime()
      if (gap > 30 * 60 * 1000) {
        totalSessions++; totalPages += sessionPages
        if (sessionPages === 1) bounceSessions++
        else { totalDurationMs += sessionLast - sessionStart; durationCount++ }
        sessionPages = 1; sessionStart = curr; sessionLast = curr
      } else { sessionPages++; sessionLast = curr }
    }
    totalSessions++; totalPages += sessionPages
    if (sessionPages === 1) bounceSessions++
    else { totalDurationMs += sessionLast - sessionStart; durationCount++ }
  })

  const sessionStats = {
    totalSessions,
    avgPages: totalSessions > 0 ? Math.round(totalPages / totalSessions * 10) / 10 : 0,
    bounceRate: totalSessions > 0 ? Math.round(bounceSessions / totalSessions * 100) : 0,
    avgSessionDuration: durationCount > 0 ? Math.round(totalDurationMs / durationCount / 1000) : 0,
  }

  // --- Streaks ---
  const activeDaySet = new Set(v.map(r => r.created_at.slice(0, 10)))
  const sortedDays = Array.from(activeDaySet).sort()
  let longestStreak = 0, streak = 1
  for (let i = 1; i < sortedDays.length; i++) {
    const diff = (new Date(sortedDays[i]).getTime() - new Date(sortedDays[i - 1]).getTime()) / 86400000
    if (diff === 1) { streak++ } else { longestStreak = Math.max(longestStreak, streak); streak = 1 }
  }
  longestStreak = Math.max(longestStreak, streak)

  const today = new Date().toISOString().slice(0, 10)
  const yesterday = new Date(Date.now() - 86400000).toISOString().slice(0, 10)
  let currentStreak = 0
  const lastDay = sortedDays[sortedDays.length - 1]
  if (lastDay === today || lastDay === yesterday) {
    currentStreak = 1
    for (let i = sortedDays.length - 2; i >= 0; i--) {
      const diff = (new Date(sortedDays[i + 1]).getTime() - new Date(sortedDays[i]).getTime()) / 86400000
      if (diff === 1) currentStreak++; else break
    }
  }
  const thisMonth = today.slice(0, 7)
  const activeDaysThisMonth = sortedDays.filter(d => d.startsWith(thisMonth)).length

  // --- Weekly new vs returning unique visitors ---
  const firstSeenWeek: Record<string, string> = {}
  v.forEach(r => {
    const week = getWeekMonday(r.created_at.slice(0, 10))
    if (!firstSeenWeek[r.visitor_id] || week < firstSeenWeek[r.visitor_id]) firstSeenWeek[r.visitor_id] = week
  })
  const weeklyVisitors: Record<string, Set<string>> = {}
  v.forEach(r => {
    const week = getWeekMonday(r.created_at.slice(0, 10))
    if (!weeklyVisitors[week]) weeklyVisitors[week] = new Set()
    weeklyVisitors[week].add(r.visitor_id)
  })
  const newVsReturning = Object.entries(weeklyVisitors)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-12)
    .map(([week, visitors]) => {
      let newCount = 0, returningCount = 0
      visitors.forEach(vid => { if (firstSeenWeek[vid] === week) newCount++; else returningCount++ })
      return { week, new: newCount, returning: returningCount }
    })

  // --- Weekly chat trend ---
  const chatByWeek: Record<string, { opens: number; queries: number }> = {}
  e.forEach(r => {
    const week = getWeekMonday(r.created_at.slice(0, 10))
    if (!chatByWeek[week]) chatByWeek[week] = { opens: 0, queries: 0 }
    if (r.event_type === 'chat_open') chatByWeek[week].opens++
    if (r.event_type === 'chat_query') chatByWeek[week].queries++
  })
  const chatTrend = Object.entries(chatByWeek)
    .sort((a, b) => a[0].localeCompare(b[0]))
    .slice(-12)
    .map(([week, d]) => ({ week, ...d }))

  return NextResponse.json({
    totalViews: v.length,
    uniqueVisitors,
    returningVisitors,
    chatOpens,
    chatQueries,
    dailyViews,
    monthlyViews,
    topPages,
    last7,
    prev7,
    last30,
    prev30,
    dayHourHeatmap,
    peakDay,
    avgDailyViews,
    deviceBreakdown,
    topCountries,
    topReferrers,
    sessionStats,
    streaks: { current: currentStreak, longest: longestStreak, activeDaysThisMonth },
    newVsReturning,
    chatTrend,
  })
}
