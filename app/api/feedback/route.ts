import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const { body } = await req.json()
    if (!body?.trim()) return NextResponse.json({ error: 'empty' }, { status: 400 })

    const res = await fetch('https://api.github.com/repos/JSam-M/movies-dashboard/issues', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/vnd.github+json',
        'Authorization': `Bearer ${process.env.GITHUB_TOKEN}`,
      },
      body: JSON.stringify({ title: 'Feedback', body: body.trim(), labels: ['feedback'] }),
    })

    if (res.ok) return NextResponse.json({ ok: true })
    return NextResponse.json({ error: 'github_error' }, { status: 500 })
  } catch {
    return NextResponse.json({ error: 'failed' }, { status: 500 })
  }
}
