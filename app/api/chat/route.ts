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

    // Fuzzy-match a bare title query (e.g. "vazha" → "Vaazha")
    const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')
    const normQuery = normalize(query)
    const titleMatch = query.trim().split(/\s+/).length <= 4
      ? allMovies.find(m => {
          const t = normalize(m.name)
          return t === normQuery || t.startsWith(normQuery) || normQuery.startsWith(t)
        })
      : null

    // Minimal catalogue — name|year|language|genre|director|rating, top 400 by rating
    const catalogue = allMovies
      .sort((a, b) => b.tmdbRating - a.tmdbRating)
      .slice(0, 400)
      .map(m => {
        const genre = m.genre.split(',')[0].trim()
        const director = m.director.split(',')[0].trim()
        const rw = m.timesWatched >= 2 ? '★' : ''
        return `${m.name}|${m.releaseYear}|${m.language}|${genre}|${director}|${m.tmdbRating}${rw}`
      }).join('\n')

    const systemPrompt = `Film recommender. Recommend ONLY from the catalogue below.
Rules:
- ONLY output films that appear in the catalogue — never name a film that is not in it.
- If the user references a film not in the catalogue, use your knowledge of that film to infer their taste, then find similar films that ARE in the catalogue. Do not mention that the reference film is absent.
- Format each pick as: **Name** (Year, Language) — one sentence why.
- Give 3-5 recommendations. ★=personally rewatched.
- Be concise. No preamble. No follow-up questions.

CATALOGUE (Name|Year|Language|Genre|Director|Rating):\n${catalogue}`

    // Send only last 4 messages to keep token count low; rewrite bare title queries
    const trimmedMessages = messages.slice(-4).map((m: {role:string; content:string}, i: number, arr: {role:string; content:string}[]) => {
      if (titleMatch && m.role === 'user' && i === arr.length - 1) {
        return { ...m, content: `I liked "${titleMatch.name}" — recommend similar films from the catalogue.` }
      }
      return m
    })

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
