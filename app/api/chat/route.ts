import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'

const client = new Anthropic()

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json()

    const allMovies = getUniqueMovies()

    // Compact catalogue — name, year, language, top 2 genres, director, rating only
    // Removes overviews entirely — cuts tokens by ~70%
    const catalogue = allMovies.map(m => {
      const genres = m.genre.split(',').slice(0,2).map((g: string) => g.trim()).join('/')
      const director = m.director.split(',')[0].trim()
      const rw = m.timesWatched >= 2 ? '★' : ''
      return `${m.name}|${m.releaseYear}|${m.language}|${genres}|${director}|${m.tmdbRating}${rw}`
    }).join('\n')

    const systemPrompt = `You are a film recommendation assistant for a personal movie collection.
${allMovies.length} films watched. Recommend ONLY from this list.

Rules:
- Give 3-5 recommendations per reply
- Format: **Film Name** (Year, Language) — one sentence reason
- If user mentions a film not in the list, use your knowledge of that film's genre/tone/themes to find the closest matches FROM THE LIST
- Be warm and specific. After recs, ask one follow-up to refine
- ★ = personally rewatched multiple times (strong endorsement)

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
