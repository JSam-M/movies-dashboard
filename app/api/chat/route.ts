import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'
import { needsApiCall } from '@/lib/heuristic'

const client = new Anthropic()

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json()

    // Only the latest user message is used for heuristic check
    const lastUserMsg = [...messages].reverse().find((m: {role:string}) => m.role === 'user')
    const query = lastUserMsg?.content || ''

    // If query doesn't need API, signal the client to use heuristic
    if (!needsApiCall(query)) {
      return NextResponse.json({ useHeuristic: true })
    }

    const allMovies = getUniqueMovies()

    // Minimal catalogue — name|genre|rating only, no overviews, top 400 by rating
    const catalogue = allMovies
      .sort((a, b) => b.tmdbRating - a.tmdbRating)
      .slice(0, 400)
      .map(m => {
        const genre = m.genre.split(',')[0].trim()
        const rw = m.timesWatched >= 2 ? '\u2605' : ''
        return `${m.name}|${m.releaseYear}|${m.language}|${genre}|${m.tmdbRating}${rw}`
      }).join('\n')

    const systemPrompt = `Film recommender. Recommend ONLY from catalogue below.
Format: **Name** (Year, Language) — one sentence why. Give 3-5 recs. \u2605=personally rewatched.
Be concise. No preamble.

CATALOGUE (Name|Year|Language|Genre|Rating):\n${catalogue}`

    // Send only last 4 messages to keep token count low
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
