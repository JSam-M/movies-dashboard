'use client'

import { useState, useEffect } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'
import type { ReactNode } from 'react'
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell,
} from 'recharts'
import ThemeToggle from '@/components/ThemeToggle'
import AboutModal from '@/components/AboutModal'

interface AnalyticsData {
  totalViews: number
  uniqueVisitors: number
  returningVisitors: number
  chatOpens: number
  chatQueries: number
  dailyViews: { date: string; count: number }[]
  topPages: { path: string; count: number }[]
  last7: number
  prev7: number
  hourlyViews: { hour: number; count: number }[]
  dowViews: { day: number; label: string; count: number }[]
  peakDay: { date: string; count: number }
  avgDailyViews: number
}

const QUAL = ['#0071e3', '#5856d6', '#34c759', '#ff9500', '#ff3b30', '#af52de', '#32ade6', '#ff2d55']

const tt = {
  contentStyle: {
    background: 'var(--modal-bg)', border: '1px solid var(--fill-border)',
    borderRadius: '12px', boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
    fontFamily: 'inherit', fontSize: '12px', padding: '10px 14px',
  },
  labelStyle: { color: 'var(--text)', fontWeight: 600 },
  itemStyle: { color: 'var(--sub)' },
  cursor: { fill: 'var(--fill)' },
}

function fmtDate(s: string) {
  const parts = s.split('-').map(Number)
  if (parts.length !== 3) return s
  const [, m, d] = parts
  return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m - 1]} ${d}`
}

function fmtDateLong(s: string) {
  const parts = s.split('-').map(Number)
  if (parts.length !== 3) return s
  const [y, m, d] = parts
  return `${['January','February','March','April','May','June','July','August','September','October','November','December'][m - 1]} ${d}, ${y}`
}

function fmtHour(h: number) {
  if (h === 0) return '12am'
  if (h === 12) return '12pm'
  return h < 12 ? `${h}am` : `${h - 12}pm`
}

function Section({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <div className="mb-14">
      <p className="font-body text-[0.6rem] font-semibold tracking-[0.14em] uppercase text-[var(--sub)] mb-1">{eyebrow}</p>
      <p className="font-display text-[1.5rem] font-light text-[var(--text)] mb-6">{title}</p>
      {children}
    </div>
  )
}

function KPICard({ value, label, sub, dot, trend, isText }: {
  value: string | number; label: string; sub?: string; dot?: string
  trend?: { pct: number; dir: 'up' | 'down' | 'flat' }; isText?: boolean
}) {
  const str = String(value)
  const long = isText ?? str.length > 6
  const trendColor = trend?.dir === 'up' ? '#34c759' : trend?.dir === 'down' ? '#ff3b30' : 'var(--muted)'
  const trendArrow = trend?.dir === 'up' ? '↑' : trend?.dir === 'down' ? '↓' : '→'

  return (
    <div
      className="glass rounded-2xl p-6 relative overflow-hidden flex flex-col"
      style={{ minHeight: '130px' }}
    >
      {dot && <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{ background: dot, opacity: 0.85 }} />}
      <div className="flex-1 flex items-start">
        {trend && (
          <div className="flex items-center gap-1 font-body text-[0.65rem] font-medium" style={{ color: trendColor }}>
            <span>{trendArrow}</span>
            <span>{trend.pct}%</span>
            <span className="font-normal" style={{ color: 'var(--muted)' }}>vs prev 7d</span>
          </div>
        )}
      </div>
      <div>
        <div className={`font-light leading-none tracking-tight text-[var(--text)] ${long ? 'font-body text-[1rem] leading-snug' : 'font-display text-[2.6rem]'}`}>
          {value}
        </div>
        <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">{label}</div>
        {sub && <div className="font-body text-[0.6rem] text-[var(--muted)] mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

function HourlyHeatmap({ data }: { data: { hour: number; count: number }[] }) {
  const max = Math.max(...data.map(d => d.count), 1)
  const labelHours = new Set([0, 6, 12, 18, 23])
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(24, 1fr)', gap: '3px', marginBottom: '6px' }}>
        {data.map(d => (
          <div
            key={d.hour}
            style={{
              height: '40px', borderRadius: '5px',
              background: '#0071e3',
              opacity: d.count === 0 ? 0.05 : 0.08 + (d.count / max) * 0.82,
              cursor: 'default',
            }}
            title={`${fmtHour(d.hour)}: ${d.count} views`}
          />
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(24, 1fr)', gap: '3px' }}>
        {data.map((d) => (
          <div key={d.hour} className="font-body text-center" style={{ fontSize: '0.55rem', color: 'var(--muted)', overflow: 'hidden', whiteSpace: 'nowrap' }}>
            {labelHours.has(d.hour) ? fmtHour(d.hour) : ''}
          </div>
        ))}
      </div>
    </div>
  )
}

function ChatFunnel({ totalViews, chatOpens, chatQueries }: {
  totalViews: number; chatOpens: number; chatQueries: number
}) {
  const openRate = totalViews > 0 ? (chatOpens / totalViews * 100).toFixed(1) : '0'
  const qps = chatOpens > 0 ? (chatQueries / chatOpens).toFixed(1) : '—'
  const maxVal = totalViews || 1
  const steps = [
    { label: 'Page Views', value: totalViews, color: '#0071e3', note: '' },
    { label: 'Chat Opens', value: chatOpens, color: '#5856d6', note: `${openRate}% of visits` },
    { label: 'Queries Sent', value: chatQueries, color: '#34c759', note: `${qps} per session` },
  ]
  return (
    <div>
      <div className="space-y-5 mb-6">
        {steps.map((s) => (
          <div key={s.label}>
            <div className="flex items-baseline justify-between mb-1.5">
              <div>
                <span className="font-body text-[0.75rem] text-[var(--sub)]">{s.label}</span>
                {s.note && <span className="font-body text-[0.65rem] text-[var(--muted)] ml-2">{s.note}</span>}
              </div>
              <span className="font-display text-[1.6rem] font-light" style={{ color: s.color }}>
                {s.value.toLocaleString()}
              </span>
            </div>
            <div className="h-[4px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${(s.value / maxVal) * 100}%`, background: s.color }}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-6 pt-5" style={{ borderTop: '1px solid var(--separator)' }}>
        <div>
          <p className="font-display text-[2.2rem] font-light" style={{ color: '#0071e3' }}>{openRate}%</p>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mt-1">Chat Rate</p>
        </div>
        <div style={{ width: '1px', height: '48px', background: 'var(--separator)' }} />
        <div>
          <p className="font-display text-[2.2rem] font-light" style={{ color: '#5856d6' }}>{qps}</p>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.1em] uppercase text-[var(--sub)] mt-1">Queries / Session</p>
        </div>
      </div>
    </div>
  )
}

function Nav({ onAbout }: { onAbout: () => void }) {
  const router = useRouter()
  return (
    <nav className="liquid-nav sticky top-0 z-40">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between" style={{ position: 'relative' }}>
        <div
          onDoubleClick={() => router.push('/analytics')}
          style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', width: '33%', height: '100%', touchAction: 'manipulation', cursor: 'default', zIndex: 0 }}
        />
        <div className="flex items-center gap-2" style={{ position: 'relative', zIndex: 1 }}>
          <Link href="/" className="flex items-center gap-2 hover:opacity-70 transition-opacity">
            <div style={{ width: '22px', height: '22px', borderRadius: '6px', background: '#0071e3', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', fontFamily: 'Georgia,serif', fontSize: '12px', fontWeight: 300, color: 'white', letterSpacing: '-0.5px', flexShrink: 0 }}>fc</div>
            <span className="font-display text-[1rem] font-light text-[var(--text)] hidden sm:inline">Film Collection</span>
          </Link>
        </div>
        <div className="flex items-center gap-3" style={{ position: 'relative', zIndex: 1 }}>
          <button onClick={onAbout} className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors">About</button>
          <Link href="/stats" className="font-body text-[0.75rem] font-medium text-[var(--sub)] hover:text-[var(--text)] transition-colors flex items-center gap-1.5">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
            <span className="hidden sm:inline">My Stats</span>
          </Link>
          <ThemeToggle />
        </div>
      </div>
    </nav>
  )
}

export default function AnalyticsPage() {
  const [password, setPassword] = useState('')
  const [authed, setAuthed]     = useState(false)
  const [error, setError]       = useState('')
  const [data, setData]         = useState<AnalyticsData | null>(null)
  const [loading, setLoading]   = useState(false)
  const [aboutOpen, setAboutOpen] = useState(false)

  useEffect(() => {
    const saved = sessionStorage.getItem('analytics_pw')
    if (saved) tryFetch(saved)
  }, [])

  async function tryFetch(pw: string) {
    setLoading(true)
    const res = await fetch('/api/analytics', { headers: { 'x-analytics-password': pw } })
    if (res.status === 401) { setError('Wrong password'); setLoading(false); return }
    const json = await res.json()
    sessionStorage.setItem('analytics_pw', pw)
    setData(json); setAuthed(true); setLoading(false)
  }

  if (!authed) return (
    <div className="min-h-screen mesh-bg flex flex-col">
      <Nav onAbout={() => setAboutOpen(true)} />
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="glass rounded-3xl p-10 w-full max-w-sm">
          <div className="mb-8">
            <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Internal</p>
            <p className="font-display text-[2.2rem] font-light text-[var(--text)] leading-none mb-3">Analytics</p>
            <p className="font-body text-[0.8rem] text-[var(--muted)] leading-relaxed">Site analytics for Film Collection. Enter the password to continue.</p>
          </div>
          <input
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && tryFetch(password)}
            placeholder="Password"
            autoFocus
            className="w-full px-4 py-3 rounded-xl font-body text-[0.9rem] text-[var(--text)] outline-none mb-3"
            style={{ background: 'var(--fill)', border: '1px solid var(--fill-border)' }}
          />
          {error && <p className="font-body text-[0.75rem] mb-3" style={{ color: '#ff3b30' }}>{error}</p>}
          <button
            onClick={() => tryFetch(password)}
            disabled={loading}
            className="w-full py-3 rounded-xl font-body text-[0.85rem] font-medium text-white transition-all"
            style={{ background: 'var(--blue)', opacity: loading ? 0.6 : 1 }}
          >
            {loading ? 'Checking…' : 'Continue'}
          </button>
        </div>
      </div>
    </div>
  )

  if (!data) return null

  const returnRate = data.uniqueVisitors > 0 ? Math.round(data.returningVisitors / data.uniqueVisitors * 100) : 0
  const chatRate = data.totalViews > 0 ? (data.chatOpens / data.totalViews * 100).toFixed(1) : '0'
  const qps = data.chatOpens > 0 ? (data.chatQueries / data.chatOpens).toFixed(1) : '—'

  const trendPct = data.prev7 > 0
    ? Math.abs(Math.round((data.last7 - data.prev7) / data.prev7 * 100))
    : data.last7 > 0 ? 100 : 0
  const trendDir: 'up' | 'down' | 'flat' = data.prev7 === 0
    ? (data.last7 > 0 ? 'up' : 'flat')
    : data.last7 > data.prev7 ? 'up' : data.last7 < data.prev7 ? 'down' : 'flat'
  const trend = { pct: trendPct, dir: trendDir }

  const withRolling = data.dailyViews.map((d, i) => {
    const window = data.dailyViews.slice(Math.max(0, i - 6), i + 1)
    const avg = window.reduce((s, w) => s + w.count, 0) / window.length
    return { ...d, avg: Math.round(avg * 10) / 10 }
  })

  const tickInterval = Math.max(1, Math.floor(withRolling.length / 10))
  const maxPage = data.topPages[0]?.count || 1
  const hasHourlyData = data.hourlyViews?.some(d => d.count > 0)

  return (
    <div className="min-h-screen mesh-bg">
      <Nav onAbout={() => setAboutOpen(true)} />
      {aboutOpen && <AboutModal onClose={() => setAboutOpen(false)} />}

      <div className="max-w-[1100px] mx-auto px-4 sm:px-8 py-12">

        {/* Header */}
        <div className="mb-12 pb-8" style={{ borderBottom: '1px solid var(--separator)' }}>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.16em] uppercase text-[var(--sub)] mb-2">Internal Dashboard</p>
          <h1 className="font-display text-[3rem] sm:text-[3.5rem] font-light text-[var(--text)] leading-none mb-3">Analytics</h1>
          <p className="font-body text-[0.85rem] text-[var(--muted)]">
            Film Collection · All-time · {data.totalViews.toLocaleString()} total views
          </p>
        </div>

        {/* KPI Grid */}
        <Section eyebrow="Overview" title="Key Metrics">
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-4">
            <KPICard
              value={data.totalViews.toLocaleString()}
              label="Total Views"
              dot="#0071e3"
              trend={trend}
            />
            <KPICard
              value={data.uniqueVisitors.toLocaleString()}
              label="Unique Visitors"
              sub={`${returnRate}% returned`}
              dot="#5856d6"
            />
            <KPICard
              value={data.avgDailyViews}
              label="Avg Daily Views"
              sub={`${data.dailyViews.length} active days`}
              dot="#ff9500"
            />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
            <KPICard
              value={data.chatOpens.toLocaleString()}
              label="Chat Opens"
              sub={`${chatRate}% of visits`}
              dot="#34c759"
            />
            <KPICard
              value={data.chatQueries.toLocaleString()}
              label="Queries Sent"
              sub={`${qps} per session`}
              dot="#af52de"
            />
            <KPICard
              value={data.peakDay.date !== '—' ? `${data.peakDay.count} views` : '—'}
              label="Peak Day"
              sub={data.peakDay.date !== '—' ? fmtDate(data.peakDay.date) : undefined}
              dot="#ff3b30"
              isText
            />
          </div>
        </Section>

        {/* Daily Trend */}
        <Section eyebrow="Traffic" title="Daily Trend">
          <div className="glass rounded-2xl p-5 sm:p-7">
            <div className="flex items-start justify-between mb-7">
              <div>
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Page Views Over Time</p>
                <p className="font-body text-[0.7rem] text-[var(--muted)] mt-1">Last {withRolling.length} days</p>
              </div>
              <div className="flex items-center gap-4 flex-shrink-0">
                <div className="flex items-center gap-2">
                  <div className="w-6 h-[2px] rounded-full" style={{ background: '#0071e3' }} />
                  <span className="font-body text-[0.65rem] text-[var(--muted)]">Daily</span>
                </div>
                <div className="flex items-center gap-2">
                  <svg width="20" height="4" viewBox="0 0 20 4"><line x1="0" y1="2" x2="20" y2="2" stroke="#ff9500" strokeWidth="1.5" strokeDasharray="4 3"/></svg>
                  <span className="font-body text-[0.65rem] text-[var(--muted)]">7-day avg</span>
                </div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={240}>
              <AreaChart data={withRolling} margin={{ left: 0, right: 8, top: 8, bottom: 44 }}>
                <defs>
                  <linearGradient id="viewsGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0071e3" stopOpacity={0.18} />
                    <stop offset="95%" stopColor="#0071e3" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" vertical={false} />
                <XAxis
                  dataKey="date"
                  tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }}
                  axisLine={false} tickLine={false}
                  angle={-45} textAnchor="end"
                  interval={tickInterval}
                  tickFormatter={fmtDate}
                />
                <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip
                  {...tt}
                  labelFormatter={fmtDateLong}
                  formatter={(val: number, name: string) => [val, name === 'count' ? 'Views' : '7-day avg']}
                />
                <Area type="monotone" dataKey="count" name="count" stroke="#0071e3" strokeWidth={2} fill="url(#viewsGrad)" dot={false} />
                <Area type="monotone" dataKey="avg" name="avg" stroke="#ff9500" strokeWidth={1.5} strokeDasharray="4 3" fill="none" fillOpacity={0} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </Section>

        {/* Patterns */}
        <Section eyebrow="Patterns" title="Behavior & Engagement">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">By Day of Week</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">UTC · weekend shown in purple</p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={data.dowViews} layout="vertical" margin={{ left: 0, right: 28, top: 0, bottom: 0 }}>
                  <XAxis type="number" tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <YAxis type="category" dataKey="label" width={36} tick={{ fontFamily: 'inherit', fontSize: 11, fill: '#86868b' }} axisLine={false} tickLine={false} />
                  <Tooltip {...tt} formatter={(v: number) => [v, 'Views']} />
                  <Bar dataKey="count" name="Views" radius={[0, 4, 4, 0]} label={{ position: 'right', fontSize: 10, fill: '#86868b', fontFamily: 'inherit' }}>
                    {data.dowViews.map((_, i) => (
                      <Cell key={i} fill={[0, 6].includes(i) ? '#5856d6' : '#0071e3'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Chat Engagement</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">Conversion from view → chat → query</p>
              <ChatFunnel totalViews={data.totalViews} chatOpens={data.chatOpens} chatQueries={data.chatQueries} />
            </div>
          </div>
        </Section>

        {/* Hourly Heatmap */}
        {hasHourlyData && (
          <Section eyebrow="Activity" title="Hourly Distribution">
            <div className="glass rounded-2xl p-5 sm:p-7">
              <div className="flex items-start justify-between mb-6">
                <div>
                  <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Views by Hour</p>
                  <p className="font-body text-[0.7rem] text-[var(--muted)] mt-1">UTC · darker = more traffic</p>
                </div>
                <div className="flex items-center gap-1.5 flex-shrink-0">
                  <div className="w-4 h-4 rounded-sm" style={{ background: '#0071e3', opacity: 0.1 }} />
                  <div className="w-4 h-4 rounded-sm" style={{ background: '#0071e3', opacity: 0.4 }} />
                  <div className="w-4 h-4 rounded-sm" style={{ background: '#0071e3', opacity: 0.7 }} />
                  <div className="w-4 h-4 rounded-sm" style={{ background: '#0071e3', opacity: 0.9 }} />
                </div>
              </div>
              <HourlyHeatmap data={data.hourlyViews} />
            </div>
          </Section>
        )}

        {/* Top Pages */}
        <Section eyebrow="Content" title="Top Pages">
          <div className="glass rounded-2xl p-5 sm:p-7">
            {data.topPages.length > 0 ? (
              <div>
                {data.topPages.map((p, i) => (
                  <div
                    key={p.path}
                    className="py-3.5"
                    style={{ borderBottom: i < data.topPages.length - 1 ? '1px solid var(--separator)' : 'none' }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3 min-w-0">
                        <span className="font-body text-[0.6rem] w-4 text-right flex-shrink-0" style={{ color: 'var(--muted)' }}>{i + 1}</span>
                        <span className="font-body text-[0.85rem] font-medium text-[var(--text)] truncate">{p.path}</span>
                      </div>
                      <div className="flex items-center gap-4 flex-shrink-0 ml-4">
                        <span className="font-body text-[0.7rem]" style={{ color: 'var(--muted)' }}>
                          {Math.round(p.count / data.totalViews * 100)}%
                        </span>
                        <span className="font-display text-[1.3rem] font-light" style={{ color: QUAL[i % QUAL.length], minWidth: '3ch', textAlign: 'right' }}>
                          {p.count.toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <div className="ml-7">
                      <div className="h-[3px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${(p.count / maxPage) * 100}%`, background: QUAL[i % QUAL.length], opacity: 0.7 }}
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="font-body text-[0.85rem] text-[var(--muted)] py-8 text-center">No data yet</p>
            )}
          </div>
        </Section>

        {/* Week-over-week detail */}
        <Section eyebrow="Recency" title="Week over Week">
          <div className="grid grid-cols-2 gap-4">
            <div className="glass rounded-2xl p-6 flex flex-col justify-between" style={{ minHeight: '110px' }}>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Last 7 Days</p>
              <div>
                <p className="font-display text-[2.6rem] font-light text-[var(--text)] leading-none">{data.last7.toLocaleString()}</p>
                <p className="font-body text-[0.6rem] text-[var(--muted)] mt-1">page views</p>
              </div>
            </div>
            <div className="glass rounded-2xl p-6 flex flex-col justify-between" style={{ minHeight: '110px' }}>
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Previous 7 Days</p>
              <div>
                <p className="font-display text-[2.6rem] font-light text-[var(--text)] leading-none">{data.prev7.toLocaleString()}</p>
                <p className="font-body text-[0.6rem] mt-1" style={{
                  color: trendDir === 'up' ? '#34c759' : trendDir === 'down' ? '#ff3b30' : 'var(--muted)',
                }}>
                  {trendDir === 'up' ? `↑ ${trendPct}% growth` : trendDir === 'down' ? `↓ ${trendPct}% decline` : 'no change'}
                </p>
              </div>
            </div>
          </div>
        </Section>

        {/* Footer */}
        <div className="pt-6 text-center pb-8" style={{ borderTop: '1px solid var(--separator)' }}>
          <p className="font-body text-[0.6rem] tracking-[0.12em] uppercase" style={{ color: 'var(--muted)' }}>
            {data.totalViews.toLocaleString()} views · {new Date().toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
          </p>
        </div>

      </div>
    </div>
  )
}
