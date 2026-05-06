import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'

const client = new Anthropic()

export async function POST(req: NextRequest) {
  try {
    const { messages, likedMovies, watchedMovies } = await req.json()

    const allMovies = getUniqueMovies()

    // Build a compact movie catalogue for the AI context
    const catalogue = allMovies.map(m =>
      `${m.name} (${m.releaseYear}, ${m.language}) | ${m.genre} | Dir: ${m.director} | TMDb: ${m.tmdbRating} | ${m.timesWatched >= 2 ? 'REWATCHED' : ''} | Synopsis: ${m.overview?.slice(0, 120)}`
    ).join('\n')

    const systemPrompt = `You are a film recommendation assistant for a personal movie collection dashboard.

The user has watched ${allMovies.length} films. Your job is to recommend films FROM THIS LIST ONLY based on their taste.

IMPORTANT RULES:
- ONLY recommend films that exist in the catalogue below
- Give 3-5 recommendations per response
- For each recommendation, give a ONE sentence reason why they'd like it based on their stated preferences
- Be conversational, warm, and specific — reference exact things from the films
- If the user mentions a film not in the catalogue, acknowledge it but pivot to what IS in the list
- Format each recommendation as: **Film Name** (Year) — reason
- After recommendations, ask a follow-up to refine further (e.g. "Want something shorter?" or "More Tamil cinema?")

${likedMovies?.length > 0 ? `User has indicated they like: ${likedMovies.join(', ')}` : ''}
${watchedMovies?.length > 0 ? `User has watched recently: ${watchedMovies.join(', ')}` : ''}

FULL FILM CATALOGUE (recommend ONLY from this list):
${catalogue}`

    const response = await client.messages.create({
      model: 'claude-sonnet-4-6',
      max_tokens: 1024,
      system: systemPrompt,
      messages: messages,
    })

    const text = response.content[0].type === 'text' ? response.content[0].text : ''
    return NextResponse.json({ message: text })

  } catch (err) {
    console.error('Chat error:', err)
    return NextResponse.json({ error: 'Failed to get recommendation' }, { status: 500 })
  }
}
