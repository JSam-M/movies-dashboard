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
  monthlyViews: { month: string; count: number }[]
  topPages: { path: string; count: number }[]
  last7: number
  prev7: number
  last30: number
  prev30: number
  dayHourHeatmap: { day: number; hour: number; count: number }[]
  peakDay: { date: string; count: number }
  avgDailyViews: number
  deviceBreakdown: { type: string; count: number }[]
  topCountries: { country: string; count: number }[]
  topReferrers: { source: string; count: number }[]
  sessionStats: { totalSessions: number; avgPages: number; bounceRate: number; avgSessionDuration: number }
  streaks: { current: number; longest: number; activeDaysThisMonth: number }
  newVsReturning: { week: string; new: number; returning: number }[]
  chatTrend: { week: string; opens: number; queries: number }[]
}

const QUAL = ['#0071e3', '#5856d6', '#34c759', '#ff9500', '#ff3b30', '#af52de', '#32ade6', '#ff2d55']
const DEVICE_COLOR: Record<string, string> = { desktop: '#0071e3', mobile: '#34c759', tablet: '#ff9500', unknown: '#86868b' }

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
  return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} ${d}`
}
function fmtDateLong(s: string) {
  const parts = s.split('-').map(Number)
  if (parts.length !== 3) return s
  const [y, m, d] = parts
  return `${['January','February','March','April','May','June','July','August','September','October','November','December'][m-1]} ${d}, ${y}`
}
function fmtHour(h: number) {
  if (h === 0) return '12a'
  if (h === 12) return '12p'
  return h < 12 ? `${h}a` : `${h-12}p`
}
function fmtMonth(s: string) {
  const [y, m] = s.split('-').map(Number)
  return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} '${String(y).slice(2)}`
}
function fmtWeek(s: string) {
  const parts = s.split('-').map(Number)
  if (parts.length !== 3) return s
  const [, m, d] = parts
  return `${['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} ${d}`
}
function fmtDuration(s: number) {
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60), sec = s % 60
  return sec > 0 ? `${m}m ${sec}s` : `${m}m`
}
function countryFlag(code: string) {
  try {
    return code.toUpperCase().split('').map(c => String.fromCodePoint(0x1F1E6 + c.charCodeAt(0) - 65)).join('')
  } catch { return '🌐' }
}
function trendCalc(current: number, prev: number) {
  const pct = prev > 0 ? Math.abs(Math.round((current - prev) / prev * 100)) : current > 0 ? 100 : 0
  const dir: 'up' | 'down' | 'flat' = prev === 0 ? (current > 0 ? 'up' : 'flat') : current > prev ? 'up' : current < prev ? 'down' : 'flat'
  return { pct, dir }
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
    <div className="glass rounded-2xl p-5 relative overflow-hidden flex flex-col" style={{ minHeight: '120px' }}>
      {dot && <div className="absolute top-4 right-4 w-2 h-2 rounded-full" style={{ background: dot, opacity: 0.85 }} />}
      <div className="flex-1 flex items-start">
        {trend && (
          <div className="flex items-center gap-1 font-body text-[0.65rem] font-medium" style={{ color: trendColor }}>
            <span>{trendArrow}</span><span>{trend.pct}%</span>
            <span className="font-normal" style={{ color: 'var(--muted)' }}>vs prev</span>
          </div>
        )}
      </div>
      <div>
        <div className={`font-light leading-none tracking-tight text-[var(--text)] ${long ? 'font-body text-[0.95rem] leading-snug' : 'font-display text-[2.4rem]'}`}>{value}</div>
        <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">{label}</div>
        {sub && <div className="font-body text-[0.6rem] text-[var(--muted)] mt-0.5">{sub}</div>}
      </div>
    </div>
  )
}

function DayHourHeatmap({ data }: { data: { day: number; hour: number; count: number }[] }) {
  const max = Math.max(...data.map(d => d.count), 1)
  const DOW = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  const labelHours = new Set([0, 6, 12, 18, 23])
  return (
    <div style={{ overflowX: 'auto', WebkitOverflowScrolling: 'touch' } as React.CSSProperties & { WebkitOverflowScrolling: string }}>
      <div style={{ minWidth: '520px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '32px repeat(24, 1fr)', gap: '2px', marginBottom: '4px' }}>
          <div />
          {Array.from({ length: 24 }, (_, h) => (
            <div key={h} style={{ fontSize: '0.5rem', color: 'var(--muted)', textAlign: 'center', whiteSpace: 'nowrap' }}>
              {labelHours.has(h) ? fmtHour(h) : ''}
            </div>
          ))}
        </div>
        {DOW.map((label, day) => {
          const row = data.filter(d => d.day === day).sort((a, b) => a.hour - b.hour)
          return (
            <div key={day} style={{ display: 'grid', gridTemplateColumns: '32px repeat(24, 1fr)', gap: '2px', marginBottom: '2px' }}>
              <div style={{ fontSize: '0.6rem', color: 'var(--muted)', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '4px' }}>{label}</div>
              {row.map(d => (
                <div
                  key={d.hour}
                  style={{
                    height: '26px', borderRadius: '4px', background: '#0071e3',
                    opacity: d.count === 0 ? 0.05 : 0.08 + (d.count / max) * 0.82,
                  }}
                  title={`${label} ${fmtHour(d.hour)}: ${d.count} views`}
                />
              ))}
            </div>
          )
        })}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '10px', justifyContent: 'flex-end' }}>
          <span style={{ fontSize: '0.55rem', color: 'var(--muted)' }}>Less</span>
          {[0.05, 0.25, 0.5, 0.75, 0.9].map(op => (
            <div key={op} style={{ width: '14px', height: '14px', borderRadius: '3px', background: '#0071e3', opacity: op }} />
          ))}
          <span style={{ fontSize: '0.55rem', color: 'var(--muted)' }}>More</span>
        </div>
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
        {steps.map(s => (
          <div key={s.label}>
            <div className="flex items-baseline justify-between mb-1.5">
              <div>
                <span className="font-body text-[0.75rem] text-[var(--sub)]">{s.label}</span>
                {s.note && <span className="font-body text-[0.65rem] text-[var(--muted)] ml-2">{s.note}</span>}
              </div>
              <span className="font-display text-[1.6rem] font-light" style={{ color: s.color }}>{s.value.toLocaleString()}</span>
            </div>
            <div className="h-[4px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
              <div className="h-full rounded-full" style={{ width: `${(s.value / maxVal) * 100}%`, background: s.color }} />
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

function RankedList({ items, label }: { items: { key: string; display: string; count: number }[]; label: string }) {
  const max = items[0]?.count || 1
  if (items.length === 0) return <p className="font-body text-[0.8rem] text-[var(--muted)] py-4 text-center">No data yet</p>
  return (
    <div className="space-y-3">
      {items.map((item, i) => (
        <div key={item.key}>
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-body text-[0.6rem] w-4 text-right flex-shrink-0" style={{ color: 'var(--muted)' }}>{i + 1}</span>
              <span className="font-body text-[0.8rem] text-[var(--text)] truncate">{item.display}</span>
            </div>
            <span className="font-display text-[1.2rem] font-light ml-3 flex-shrink-0" style={{ color: QUAL[i % QUAL.length] }}>
              {item.count.toLocaleString()}
            </span>
          </div>
          <div className="ml-6 h-[3px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
            <div className="h-full rounded-full" style={{ width: `${item.count / max * 100}%`, background: QUAL[i % QUAL.length], opacity: 0.7 }} />
          </div>
        </div>
      ))}
      <p className="font-body text-[0.6rem] text-[var(--muted)] pt-1">{label}</p>
    </div>
  )
}

function DeviceBreakdown({ data }: { data: { type: string; count: number }[] }) {
  const total = data.reduce((s, d) => s + d.count, 0) || 1
  const icons: Record<string, string> = { desktop: '🖥', mobile: '📱', tablet: '⬜', unknown: '?' }
  if (data.length === 0 || (data.length === 1 && data[0].type === 'unknown'))
    return <p className="font-body text-[0.8rem] text-[var(--muted)] py-4 text-center">No data yet — run migration first</p>
  return (
    <div className="space-y-4">
      {data.map(d => (
        <div key={d.type}>
          <div className="flex items-center justify-between mb-1.5">
            <span className="font-body text-[0.75rem] text-[var(--sub)] capitalize">{icons[d.type] ?? ''} {d.type}</span>
            <div className="flex items-center gap-3">
              <span className="font-body text-[0.7rem] text-[var(--muted)]">{d.count.toLocaleString()}</span>
              <span className="font-display text-[1.4rem] font-light" style={{ color: DEVICE_COLOR[d.type] ?? '#86868b', minWidth: '3.5ch', textAlign: 'right' }}>
                {Math.round(d.count / total * 100)}%
              </span>
            </div>
          </div>
          <div className="h-[4px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
            <div className="h-full rounded-full" style={{ width: `${d.count / total * 100}%`, background: DEVICE_COLOR[d.type] ?? '#86868b' }} />
          </div>
        </div>
      ))}
    </div>
  )
}

function Nav({ onAbout }: { onAbout: () => void }) {
  const router = useRouter()
  return (
    <nav className="liquid-nav sticky top-0 z-40">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-8 h-14 flex items-center justify-between" style={{ position: 'relative' }}>
        <div onDoubleClick={() => router.push('/analytics')} style={{ position: 'absolute', left: '50%', transform: 'translateX(-50%)', width: '33%', height: '100%', touchAction: 'manipulation', cursor: 'default', zIndex: 0 }} />
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
            type="password" value={password} onChange={e => setPassword(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && tryFetch(password)}
            placeholder="Password" autoFocus
            className="w-full px-4 py-3 rounded-xl font-body text-[0.9rem] text-[var(--text)] outline-none mb-3"
            style={{ background: 'var(--fill)', border: '1px solid var(--fill-border)' }}
          />
          {error && <p className="font-body text-[0.75rem] mb-3" style={{ color: '#ff3b30' }}>{error}</p>}
          <button onClick={() => tryFetch(password)} disabled={loading}
            className="w-full py-3 rounded-xl font-body text-[0.85rem] font-medium text-white transition-all"
            style={{ background: 'var(--blue)', opacity: loading ? 0.6 : 1 }}>
            {loading ? 'Checking…' : 'Continue'}
          </button>
        </div>
      </div>
    </div>
  )

  if (!data) return null

  const returnRate   = data.uniqueVisitors > 0 ? Math.round(data.returningVisitors / data.uniqueVisitors * 100) : 0
  const chatRate     = data.totalViews > 0 ? (data.chatOpens / data.totalViews * 100).toFixed(1) : '0'
  const qps          = data.chatOpens > 0 ? (data.chatQueries / data.chatOpens).toFixed(1) : '—'
  const trend7       = trendCalc(data.last7, data.prev7)
  const trend30      = trendCalc(data.last30, data.prev30)

  const withRolling = data.dailyViews.map((d, i) => {
    const window = data.dailyViews.slice(Math.max(0, i - 6), i + 1)
    const avg = window.reduce((s, w) => s + w.count, 0) / window.length
    return { ...d, avg: Math.round(avg * 10) / 10 }
  })
  const tickInterval = Math.max(1, Math.floor(withRolling.length / 10))
  const maxPage = data.topPages[0]?.count || 1

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

        {/* ── Key Metrics ── */}
        <Section eyebrow="Overview" title="Key Metrics">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-4">
            <KPICard value={data.totalViews.toLocaleString()} label="Total Views" dot="#0071e3" trend={trend7} />
            <KPICard value={data.uniqueVisitors.toLocaleString()} label="Unique Visitors" sub={`${returnRate}% returned`} dot="#5856d6" />
            <KPICard value={data.sessionStats.totalSessions.toLocaleString()} label="Total Sessions" sub={`${data.sessionStats.avgPages} pages/session`} dot="#34c759" />
            <KPICard value={data.avgDailyViews} label="Avg Daily Views" sub={`${data.dailyViews.length} active days`} dot="#ff9500" />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <KPICard value={data.chatOpens.toLocaleString()} label="Chat Opens" sub={`${chatRate}% of visits`} dot="#34c759" />
            <KPICard value={data.chatQueries.toLocaleString()} label="Queries Sent" sub={`${qps} per session`} dot="#af52de" />
            <KPICard value={`${data.sessionStats.bounceRate}%`} label="Bounce Rate" sub={data.sessionStats.avgSessionDuration > 0 ? `avg ${fmtDuration(data.sessionStats.avgSessionDuration)} session` : undefined} dot="#ff3b30" />
            <KPICard value={data.peakDay.date !== '—' ? `${data.peakDay.count} views` : '—'} label="Peak Day" sub={data.peakDay.date !== '—' ? fmtDate(data.peakDay.date) : undefined} dot="#ff2d55" isText />
          </div>
        </Section>

        {/* ── Growth ── */}
        <Section eyebrow="Traffic" title="Growth">
          {/* Daily trend */}
          <div className="glass rounded-2xl p-5 sm:p-7 mb-6">
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 mb-7">
              <div>
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Daily Views — Last {withRolling.length} Days</p>
                <p className="font-body text-[0.7rem] text-[var(--muted)] mt-1">with 7-day rolling average</p>
              </div>
              <div className="flex items-center gap-4">
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
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={withRolling} margin={{ left: 0, right: 8, top: 8, bottom: 44 }}>
                <defs>
                  <linearGradient id="viewsGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#0071e3" stopOpacity={0.18} />
                    <stop offset="95%" stopColor="#0071e3" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" vertical={false} />
                <XAxis dataKey="date" tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} angle={-45} textAnchor="end" interval={tickInterval} tickFormatter={fmtDate} />
                <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip {...tt} labelFormatter={fmtDateLong} formatter={(val: number, name: string) => [val, name === 'count' ? 'Views' : '7-day avg']} />
                <Area type="monotone" dataKey="count" name="count" stroke="#0071e3" strokeWidth={2} fill="url(#viewsGrad)" dot={false} />
                <Area type="monotone" dataKey="avg" name="avg" stroke="#ff9500" strokeWidth={1.5} strokeDasharray="4 3" fill="none" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Monthly + comparison */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
            <div className="glass rounded-2xl p-5 sm:p-6 sm:col-span-2">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Monthly Views</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">Last 12 months</p>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={data.monthlyViews} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" vertical={false} />
                  <XAxis dataKey="month" tickFormatter={fmtMonth} tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip {...tt} formatter={(v: number) => [v, 'Views']} labelFormatter={fmtMonth} />
                  <Bar dataKey="count" name="Views" radius={[4, 4, 0, 0]}>
                    {data.monthlyViews.map((_, i) => (
                      <Cell key={i} fill="#0071e3" opacity={i === data.monthlyViews.length - 1 ? 0.5 : 0.85} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="flex flex-col gap-4">
              <div className="glass rounded-2xl p-5 flex flex-col justify-between flex-1">
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Last 7 Days</p>
                <div>
                  <p className="font-display text-[2rem] sm:text-[2.4rem] font-light text-[var(--text)] leading-none">{data.last7.toLocaleString()}</p>
                  <p className="font-body text-[0.6rem] mt-1" style={{ color: trend7.dir === 'up' ? '#34c759' : trend7.dir === 'down' ? '#ff3b30' : 'var(--muted)' }}>
                    {trend7.dir === 'up' ? `↑ ${trend7.pct}%` : trend7.dir === 'down' ? `↓ ${trend7.pct}%` : '→ no change'} vs prev 7d
                  </p>
                </div>
              </div>
              <div className="glass rounded-2xl p-5 flex flex-col justify-between flex-1">
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Last 30 Days</p>
                <div>
                  <p className="font-display text-[2rem] sm:text-[2.4rem] font-light text-[var(--text)] leading-none">{data.last30.toLocaleString()}</p>
                  <p className="font-body text-[0.6rem] mt-1" style={{ color: trend30.dir === 'up' ? '#34c759' : trend30.dir === 'down' ? '#ff3b30' : 'var(--muted)' }}>
                    {trend30.dir === 'up' ? `↑ ${trend30.pct}%` : trend30.dir === 'down' ? `↓ ${trend30.pct}%` : '→ no change'} vs prev 30d
                  </p>
                </div>
              </div>
            </div>
          </div>
        </Section>

        {/* ── Visitors ── */}
        <Section eyebrow="Audience" title="Visitor Trends">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">New vs Returning</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-4">Unique visitors per week</p>
              {data.newVsReturning.length > 0 ? (
                <>
                  <div className="flex items-center gap-4 mb-4">
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm" style={{ background: '#0071e3' }} /><span className="font-body text-[0.65rem] text-[var(--muted)]">New</span></div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm" style={{ background: '#5856d6' }} /><span className="font-body text-[0.65rem] text-[var(--muted)]">Returning</span></div>
                  </div>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={data.newVsReturning} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" vertical={false} />
                      <XAxis dataKey="week" tickFormatter={fmtWeek} tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                      <Tooltip {...tt} labelFormatter={fmtWeek} formatter={(v: number, name: string) => [v, name === 'new' ? 'New' : 'Returning']} />
                      <Bar dataKey="new" name="new" stackId="a" fill="#0071e3" />
                      <Bar dataKey="returning" name="returning" stackId="a" fill="#5856d6" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <p className="font-body text-[0.8rem] text-[var(--muted)] py-8 text-center">No data yet</p>
              )}
            </div>

            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Devices</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">How visitors browse</p>
              <DeviceBreakdown data={data.deviceBreakdown ?? []} />
            </div>
          </div>
        </Section>

        {/* ── Audience — Countries + Referrers ── */}
        <Section eyebrow="Origin" title="Where Visitors Come From">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Top Countries</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">by page views</p>
              <RankedList
                label="Run migration to enable country tracking"
                items={(data.topCountries ?? []).map(c => ({
                  key: c.country,
                  display: `${countryFlag(c.country)} ${c.country}`,
                  count: c.count,
                }))}
              />
            </div>
            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Traffic Sources</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">referrer hostname · Direct = no referrer</p>
              <RankedList
                label="Run migration to enable referrer tracking"
                items={(data.topReferrers ?? []).map(r => ({
                  key: r.source,
                  display: r.source,
                  count: r.count,
                }))}
              />
            </div>
          </div>
        </Section>

        {/* ── Activity — Day × Hour heatmap ── */}
        <Section eyebrow="Patterns" title="Activity Heatmap">
          <div className="glass rounded-2xl p-5 sm:p-7">
            <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-6">
              <div>
                <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)]">Views by Day & Hour</p>
                <p className="font-body text-[0.7rem] text-[var(--muted)] mt-1">UTC · darker = more traffic</p>
              </div>
            </div>
            <DayHourHeatmap data={data.dayHourHeatmap} />
          </div>
        </Section>

        {/* ── Chat ── */}
        <Section eyebrow="Engagement" title="Chat Analytics">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Conversion Funnel</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-5">View → Chat → Query</p>
              <ChatFunnel totalViews={data.totalViews} chatOpens={data.chatOpens} chatQueries={data.chatQueries} />
            </div>
            <div className="glass rounded-2xl p-5 sm:p-6">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-1">Weekly Chat Trend</p>
              <p className="font-body text-[0.7rem] text-[var(--muted)] mb-4">Opens and queries per week</p>
              {data.chatTrend.length > 0 ? (
                <>
                  <div className="flex items-center gap-4 mb-4">
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm" style={{ background: '#5856d6' }} /><span className="font-body text-[0.65rem] text-[var(--muted)]">Opens</span></div>
                    <div className="flex items-center gap-2"><div className="w-3 h-3 rounded-sm" style={{ background: '#34c759' }} /><span className="font-body text-[0.65rem] text-[var(--muted)]">Queries</span></div>
                  </div>
                  <ResponsiveContainer width="100%" height={180}>
                    <BarChart data={data.chatTrend} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" vertical={false} />
                      <XAxis dataKey="week" tickFormatter={fmtWeek} tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                      <Tooltip {...tt} labelFormatter={fmtWeek} formatter={(v: number, name: string) => [v, name === 'opens' ? 'Opens' : 'Queries']} />
                      <Bar dataKey="opens" name="opens" fill="#5856d6" radius={[2, 2, 0, 0]} />
                      <Bar dataKey="queries" name="queries" fill="#34c759" radius={[2, 2, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </>
              ) : (
                <p className="font-body text-[0.8rem] text-[var(--muted)] py-8 text-center">No data yet</p>
              )}
            </div>
          </div>
        </Section>

        {/* ── Top Pages ── */}
        <Section eyebrow="Content" title="Top Pages">
          <div className="glass rounded-2xl p-5 sm:p-7">
            {data.topPages.length > 0 ? (
              <div>
                {data.topPages.map((p, i) => (
                  <div key={p.path} className="py-3.5" style={{ borderBottom: i < data.topPages.length - 1 ? '1px solid var(--separator)' : 'none' }}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-3 min-w-0">
                        <span className="font-body text-[0.6rem] w-4 text-right flex-shrink-0" style={{ color: 'var(--muted)' }}>{i + 1}</span>
                        <span className="font-body text-[0.85rem] font-medium text-[var(--text)] truncate">{p.path}</span>
                      </div>
                      <div className="flex items-center gap-4 flex-shrink-0 ml-4">
                        <span className="font-body text-[0.7rem]" style={{ color: 'var(--muted)' }}>{Math.round(p.count / data.totalViews * 100)}%</span>
                        <span className="font-display text-[1.3rem] font-light" style={{ color: QUAL[i % QUAL.length], minWidth: '3ch', textAlign: 'right' }}>{p.count.toLocaleString()}</span>
                      </div>
                    </div>
                    <div className="ml-7">
                      <div className="h-[3px] rounded-full overflow-hidden" style={{ background: 'var(--fill)' }}>
                        <div className="h-full rounded-full" style={{ width: `${(p.count / maxPage) * 100}%`, background: QUAL[i % QUAL.length], opacity: 0.7 }} />
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

        {/* ── Records & Streaks ── */}
        <Section eyebrow="Records" title="Streaks & Milestones">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="glass rounded-2xl p-5">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-3">Current Streak</p>
              <p className="font-display text-[2.4rem] font-light text-[var(--text)] leading-none">{data.streaks.current}</p>
              <p className="font-body text-[0.6rem] text-[var(--muted)] mt-1">active days</p>
            </div>
            <div className="glass rounded-2xl p-5">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-3">Longest Streak</p>
              <p className="font-display text-[2.4rem] font-light text-[var(--text)] leading-none">{data.streaks.longest}</p>
              <p className="font-body text-[0.6rem] text-[var(--muted)] mt-1">consecutive days</p>
            </div>
            <div className="glass rounded-2xl p-5">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-3">This Month</p>
              <p className="font-display text-[2.4rem] font-light text-[var(--text)] leading-none">{data.streaks.activeDaysThisMonth}</p>
              <p className="font-body text-[0.6rem] text-[var(--muted)] mt-1">active days</p>
            </div>
            <div className="glass rounded-2xl p-5">
              <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-3">Peak Day</p>
              <p className="font-body text-[1rem] font-light text-[var(--text)] leading-snug">{data.peakDay.count > 0 ? `${data.peakDay.count} views` : '—'}</p>
              <p className="font-body text-[0.6rem] text-[var(--muted)] mt-1">{data.peakDay.date !== '—' ? fmtDateLong(data.peakDay.date) : 'no data'}</p>
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
