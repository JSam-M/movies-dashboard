import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'

const client = new Anthropic()

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json()

    const allMovies = getUniqueMovies()

    // Compact catalogue — sorted by rating desc, top 500 only
    // Last name only for director, single genre, no overviews
    const catalogue = allMovies
      .sort((a, b) => b.tmdbRating - a.tmdbRating)
      .slice(0, 500)
      .map(m => {
        const genre   = m.genre.split(',')[0].trim()
        const dirParts = m.director.split(',')[0].trim().split(' ')
        const director = dirParts[dirParts.length - 1]
        const rw = m.timesWatched >= 2 ? '★' : ''
        return `${m.name}|${m.releaseYear}|${m.language}|${genre}|${director}|${m.tmdbRating}${rw}`
      }).join('\n')

    const systemPrompt = `Film recommender. ${allMovies.length} films. Recommend ONLY from catalogue.
Format: **Name** (Year, Language) — one sentence why. Give 3-5 recs. ★=personally rewatched.
If user mentions a film not in catalogue, use your knowledge of its genre/tone to find matches FROM THE LIST.
Ask one follow-up after recs.

CATALOGUE (Name|Year|Language|Genre|Director|Rating★):
${catalogue}`

    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 600,
      system: systemPrompt,
      messages: messages,
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
