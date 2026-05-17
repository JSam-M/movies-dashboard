import { NextRequest, NextResponse } from 'next/server'
import { rateLimit } from '@/lib/rateLimit'

export async function POST(req: NextRequest) {
  try {
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0].trim() || 'unknown'
    const { allowed } = rateLimit(`feedback:${ip}`, { limit: 3, windowMs: 60_000 })
    if (!allowed) return NextResponse.json({ error: 'Too many requests' }, { status: 429 })

    const { body } = await req.json()
    if (!body?.trim()) return NextResponse.json({ error: 'empty' }, { status: 400 })

    // Truncate to prevent abuse
    const sanitized = body.trim().slice(0, 2000)

    const res = await fetch('https://api.github.com/repos/JSam-M/movies-dashboard/issues', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github+json',
        'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`,
      },
      body: JSON.stringify({ title: 'Feedback', body: sanitized, labels: ['feedback'] }),
    })

    if (res.ok) return NextResponse.json({ ok: true })
    return NextResponse.json({ error: 'github_error' }, { status: 500 })
  } catch {
    return NextResponse.json({ error: 'failed' }, { status: 500 })
  }
}
