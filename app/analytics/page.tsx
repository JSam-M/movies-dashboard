'use client'

import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'

interface AnalyticsData {
  totalViews: number
  uniqueVisitors: number
  returningVisitors: number
  chatOpens: number
  chatQueries: number
  dailyViews: { date: string; count: number }[]
  topPages: { path: string; count: number }[]
}

const tt = {
  contentStyle: { background:'var(--modal-bg)', border:'1px solid var(--fill-border)', borderRadius:'12px', fontFamily:'inherit', fontSize:'12px' },
  labelStyle: { color:'var(--text)', fontWeight:500 },
  itemStyle: { color:'var(--text)' },
  cursor: { fill:'var(--fill)' },
}

function KPI({ value, label, sub }: { value: string | number; label: string; sub?: string }) {
  return (
    <div className="glass rounded-2xl p-6" style={{ height: '120px', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
      <div className="font-display text-[2.4rem] font-light leading-none text-[var(--text)]">{value}</div>
      <div className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mt-2">{label}</div>
      {sub && <div className="font-body text-[0.6rem] text-[var(--muted)] mt-0.5">{sub}</div>}
    </div>
  )
}

export default function AnalyticsPage() {
  const [password, setPassword] = useState('')
  const [authed, setAuthed]     = useState(false)
  const [error, setError]       = useState('')
  const [data, setData]         = useState<AnalyticsData | null>(null)
  const [loading, setLoading]   = useState(false)

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
    <div className="min-h-screen mesh-bg flex items-center justify-center">
      <div className="glass rounded-3xl p-10 w-full max-w-sm">
        <p className="font-display text-[1.8rem] font-light text-[var(--text)] mb-2">Analytics</p>
        <p className="font-body text-[0.8rem] text-[var(--sub)] mb-8">Enter password to continue</p>
        <input
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && tryFetch(password)}
          placeholder="Password"
          className="w-full px-4 py-3 rounded-xl font-body text-[0.9rem] text-[var(--text)] outline-none mb-3"
          style={{ background: 'var(--fill)', border: '1px solid var(--fill-border)' }}
        />
        {error && <p className="font-body text-[0.75rem] text-red-400 mb-3">{error}</p>}
        <button onClick={() => tryFetch(password)} disabled={loading}
          className="w-full py-3 rounded-xl font-body text-[0.85rem] font-medium text-white transition-all"
          style={{ background: 'var(--blue)', opacity: loading ? 0.6 : 1 }}>
          {loading ? 'Checking…' : 'Continue'}
        </button>
      </div>
    </div>
  )

  if (!data) return null

  const queriesPerOpen = data.chatOpens > 0 ? (data.chatQueries / data.chatOpens).toFixed(1) : '—'

  return (
    <div className="min-h-screen mesh-bg">
      <div className="max-w-[1100px] mx-auto px-4 sm:px-8 py-12">

        <div className="mb-10 pb-8" style={{ borderBottom: '1px solid var(--separator)' }}>
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-2">Dashboard</p>
          <h1 className="font-display text-[2.8rem] font-light text-[var(--text)]">Analytics</h1>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 mb-10">
          <KPI value={data.totalViews}       label="Total Page Views" />
          <KPI value={data.uniqueVisitors}   label="Unique Visitors" />
          <KPI value={data.returningVisitors} label="Returning Visitors" sub="visited more than once" />
          <KPI value={data.chatOpens}        label="Chat Opens" />
          <KPI value={data.chatQueries}      label="Total Queries" />
          <KPI value={queriesPerOpen}        label="Queries per Session" />
        </div>

        {/* Daily chart */}
        <div className="glass rounded-2xl p-6 mb-10">
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-5">Daily Page Views</p>
          {data.dailyViews.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={data.dailyViews} margin={{ left: 0, right: 8, top: 4, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--separator)" />
                <XAxis dataKey="date" tick={{ fontFamily: 'inherit', fontSize: 9, fill: '#86868b' }} axisLine={false} tickLine={false} angle={-45} textAnchor="end" interval="preserveStartEnd" />
                <YAxis tick={{ fontFamily: 'inherit', fontSize: 10, fill: '#86868b' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip {...tt} />
                <Line type="monotone" dataKey="count" name="Views" stroke="#0071e3" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="font-body text-[0.85rem] text-[var(--muted)] py-8 text-center">No data yet</p>
          )}
        </div>

        {/* Top pages */}
        <div className="glass rounded-2xl p-6">
          <p className="font-body text-[0.6rem] font-semibold tracking-[0.12em] uppercase text-[var(--sub)] mb-5">Top Pages</p>
          <div className="space-y-2">
            {data.topPages.map(p => (
              <div key={p.path} className="flex items-center justify-between py-2" style={{ borderBottom: '1px solid var(--separator)' }}>
                <span className="font-body text-[0.85rem] text-[var(--text)]">{p.path}</span>
                <span className="font-body text-[0.8rem] text-[var(--muted)]">{p.count} views</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  )
}
