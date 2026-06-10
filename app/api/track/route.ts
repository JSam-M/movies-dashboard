import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'
import { rateLimit } from '@/lib/rateLimit'

function getDeviceType(ua: string): 'mobile' | 'tablet' | 'desktop' {
  if (/iPad|Tablet|PlayBook/i.test(ua)) return 'tablet'
  if (/Mobile|Android|iPhone|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua)) return 'mobile'
  return 'desktop'
}

export async function POST(req: NextRequest) {
  try {
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0].trim() || 'unknown'
    const { allowed } = rateLimit(`track:${ip}`, { limit: 30, windowMs: 60_000 })
    if (!allowed) return NextResponse.json({ ok: false }, { status: 429 })

    const { event, path, visitorId, referrer } = await req.json()
    const vid = visitorId || 'anonymous'

    if (event === 'page_view') {
      const ua = req.headers.get('user-agent') || ''
      const country = req.headers.get('x-vercel-ip-country') || null
      const device_type = getDeviceType(ua)
      await supabase.from('page_views').insert({
        path: path || '/',
        visitor_id: vid,
        device_type,
        country,
        referrer: referrer || null,
      })
    } else if (event === 'chat_open' || event === 'chat_query') {
      await supabase.from('chat_events').insert({ event_type: event, visitor_id: vid })
    }

    return NextResponse.json({ ok: true })
  } catch {
    return NextResponse.json({ ok: false }, { status: 500 })
  }
}
