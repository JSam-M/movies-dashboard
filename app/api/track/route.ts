import { NextRequest, NextResponse } from 'next/server'
import { supabase } from '@/lib/supabase'

async function hashIp(ip: string): Promise<string> {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(ip))
  return Array.from(new Uint8Array(buf)).map(b => b.toString(16).padStart(2, '0')).join('')
}

export async function POST(req: NextRequest) {
  try {
    const { event, path } = await req.json()
    const ip = req.headers.get('x-forwarded-for')?.split(',')[0].trim()
      || req.headers.get('x-real-ip')
      || '0.0.0.0'
    const ip_hash = await hashIp(ip)

    if (event === 'page_view') {
      await supabase.from('page_views').insert({ path: path || '/', ip_hash })
    } else if (event === 'chat_open' || event === 'chat_query') {
      await supabase.from('chat_events').insert({ event_type: event, ip_hash })
    }

    return NextResponse.json({ ok: true })
  } catch {
    return NextResponse.json({ ok: false }, { status: 500 })
  }
}
