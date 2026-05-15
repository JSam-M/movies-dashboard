import { NextRequest, NextResponse } from 'next/server'
import Anthropic from '@anthropic-ai/sdk'
import { getUniqueMovies } from '@/lib/movies'

const client = new Anthropic()

type Movie = ReturnType<typeof getUniqueMovies>[0]

const normalize = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, '')

function editDist(a: string, b: string): number {
  const m = a.length, n = b.length
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  )
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++)
      dp[i][j] = a[i-1] === b[j-1] ? dp[i-1][j-1] : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
  return dp[m][n]
}

function formatEntry(m: Movie): string {
  const genre = m.genre.split(',')[0].trim()
  const director = m.director.split(',')[0].trim()
  const rw = m.timesWatched >= 2 ? '★' : ''
  return `${m.name}|${m.releaseYear}|${m.language}|${genre}|${director}|${m.tmdbRating}${rw}`
}

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json()

    const lastUserMsg = [...messages].reverse().find((m: {role:string}) => m.role === 'user')
    const query = lastUserMsg?.content || ''

    const allMovies = getUniqueMovies()

    const rewrittenMessages = [...messages]

    // Fuzzy-match a short bare query against film titles (first word vs first word, edit dist ≤ 1)
    let referencedFilm: Movie | null = null
    const words = query.trim().split(/\s+/)
    if (words.length <= 4) {
      const qWord = normalize(words[0])
      if (qWord.length >= 3) {
        const matches = allMovies.filter(m => {
          const tWord = normalize(m.name.split(/[\s:–,]/)[0])
          return editDist(qWord, tWord) <= 1
        })

        if (matches.length >= 2) {
          return NextResponse.json({
            disambiguate: matches.slice(0, 4).map(m => ({
              name: m.name, year: m.releaseYear, language: m.language,
            })),
          })
        }

        if (matches.length === 1) {
          referencedFilm = matches[0]
          const idx = messages.map((m: {role:string}) => m.role).lastIndexOf('user')
          if (idx !== -1) {
            messages[idx] = {
              ...messages[idx],
              content: `I liked "${referencedFilm.name}" — recommend similar films from the catalogue.`,
            }
          }
        }
      }
    }

    // Also detect the film name from a pill-rewritten message like: I liked "Film Name" —
    if (!referencedFilm) {
      const pillMatch = query.match(/I liked "([^"]+)"/)
      if (pillMatch) {
        referencedFilm = allMovies.find(m => m.name === pillMatch[1]) ?? null
      }
    }

    // Top 400 by rating; always append films that fall outside top 400 but are relevant to the query
    const top400 = [...allMovies].sort((a, b) => b.tmdbRating - a.tmdbRating).slice(0, 400)
    const inTop400 = new Set(top400.map(m => m.name))

    const extras: Movie[] = []

    // Append the referenced film if it falls outside top 400
    if (referencedFilm && !inTop400.has(referencedFilm.name)) extras.push(referencedFilm)

    // Detect director names in the query and append all their films outside top 400
    const qLower = query.toLowerCase()
    const allDirectors = Array.from(new Set(
      allMovies.flatMap(m => m.director.split(',').map(d => d.trim()).filter(Boolean))
    ))
    const mentionedDirectors = allDirectors.filter(d => qLower.includes(d.toLowerCase()))
    if (mentionedDirectors.length) {
      allMovies
        .filter(m => mentionedDirectors.some(d => m.director.split(',').map(x => x.trim()).includes(d)))
        .filter(m => !inTop400.has(m.name) && !extras.find(e => e.name === m.name))
        .forEach(m => extras.push(m))
    }

    const catalogue = [...top400, ...extras].map(formatEntry).join('\n')

    const systemPrompt = `Film recommender. Recommend ONLY from the catalogue below.
Rules:
- ONLY output films that appear in the catalogue — never name a film not in it.
- If the user references a film not in the catalogue, use your knowledge of that film to infer their taste, then find thematically similar films that ARE in the catalogue. Do not mention the reference film is absent.
- Start with exactly one short intro line referencing what the user liked or asked for (e.g. "Since you enjoyed X, you might like these:" or "For a feel-good night, here are some picks:"). Then list recommendations.
- If the user mentions a language (e.g. "English", "Tamil"), treat it as a filter and ONLY recommend films where the Language field matches that language exactly. Do not recommend films in other languages.
- Format EXACTLY as: **Name** (Year, Language) — one sentence explaining why it suits the user's taste (themes, tone, style). Do NOT just list the genre or rating.
- Give 3-5 recommendations. ★=personally rewatched.
- End with exactly one short closing line inviting refinement (e.g. "Want me to narrow it down by mood or language?").

CATALOGUE (Name|Year|Language|Genre|Director|Rating):\n${catalogue}`

    const trimmedMessages = rewrittenMessages.slice(-4)

    const response = await client.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 500,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      system: [{ type: 'text', text: systemPrompt, cache_control: { type: 'ephemeral' } }] as any,
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
