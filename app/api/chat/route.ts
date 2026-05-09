import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'
import { needsApiCall, heuristicRecommend } from '@/lib/heuristic'

const client = new Anthropic()

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json()

    const allMovies = getUniqueMovies()

    // Only the last user message matters for routing
    const lastUserMsg = [...messages].reverse().find((m: {role:string;content:string}) => m.role === 'user')
    const lastQuery   = lastUserMsg?.content || ''

    // ── Heuristic path (free, instant) ──
    if (!needsApiCall(lastQuery)) {
      const result = heuristicRecommend(lastQuery, allMovies)
      return NextResponse.json({ message: result })
    }

    // ── API fallback (complex / similarity queries) ──
    // Send only name|genre|rating to keep token count minimal
    const catalogue = allMovies
      .sort((a, b) => b.tmdbRating - a.tmdbRating)
      .slice(0, 500)
      .map(m => {
        const genre = m.genre.split(',')[0].trim()
        const rw    = m.timesWatched >= 2 ? '★' : ''
        return `${m.name}|${m.releaseYear}|${m.language}|${genre}|${m.tmdbRating}${rw}`
      }).join('\n')

    const systemPrompt = `Film recommender. Recommend ONLY from the catalogue below.
Format each rec as: **Name** (Year, Language) — one sentence why. Give 3-5 recs. ★=personally rewatched.
If user references a film not in the catalogue, use its genre/tone to find matches FROM THE LIST.
End with one short follow-up question.

CATALOGUE (Name|Year|Language|Genre|Rating★):
${catalogue}`

    // Only send last 4 messages to keep context small
    const trimmedMessages = messages.slice(-4)

    const response = await client.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 300,
      system: systemPrompt,
      messages: trimmedMessages,
    })

    const text = response.content[0].type === 'text' ? response.content[0].text : ''
    return NextResponse.json({ message: text })

  } catch (err: unknown) {
    console.error('Chat error:', err)
    const msg = err instanceof Error ? err.message : ''
    if (msg.includes('overloaded') || msg.includes('529')) {
      return NextResponse.json({ error: 'overloaded' }, { status: 503 })
    }
    return NextResponse.json({ error: 'failed' }, { status: 500 })
  }
}
