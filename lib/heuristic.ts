// Heuristic recommender — runs client-side, no API call needed for common queries.
// needsApiCall() returns true for queries that require Claude fallback.

export interface HeuristicMovie {
  name: string
  language: string
  genre: string
  director: string
  tmdbRating: number
  releaseYear: number
  timesWatched: number
  runtime: string
  overview: string
}

interface ScoredMovie extends HeuristicMovie { score: number; reason: string }

const MOOD_MAP: Record<string, { genres: string[]; keywords: string[]; boost: number }> = {
  'feel-good':     { genres: ['Comedy','Romance','Animation','Family'], keywords: ['feel-good','feelgood','happy','light','fun','cheerful','uplifting','warm','heartwarming','comfort','comforting'], boost: 3 },
  'light-hearted': { genres: ['Comedy','Romance','Animation','Family'], keywords: ['light','lighthearted','light-hearted','breezy','easy','casual'], boost: 3 },
  'funny':         { genres: ['Comedy'], keywords: ['funny','hilarious','laugh','humor','humour','witty','comedy'], boost: 3 },
  'romantic':      { genres: ['Romance'], keywords: ['romance','romantic','love','relationship','couple'], boost: 3 },
  'sad':           { genres: ['Drama'], keywords: ['sad','cry','emotional','moving','tearjerker'], boost: 2 },
  'thriller':      { genres: ['Thriller','Crime','Mystery'], keywords: ['thriller','thrilling','suspense','tense','tension','mystery','crime'], boost: 3 },
  'action':        { genres: ['Action','Adventure'], keywords: ['action','exciting','adventure','fast','intense'], boost: 3 },
  'scary':         { genres: ['Horror'], keywords: ['horror','scary','frightening','terrifying','creepy'], boost: 3 },
  'sci-fi':        { genres: ['Science Fiction','Sci-Fi'], keywords: ['sci-fi','scifi','science fiction','space','future','futuristic'], boost: 3 },
  'animated':      { genres: ['Animation'], keywords: ['animated','animation','cartoon'], boost: 3 },
  'family':        { genres: ['Family','Animation'], keywords: ['family','kids','children','all ages'], boost: 3 },
  'documentary':   { genres: ['Documentary'], keywords: ['documentary','doc','real','true story'], boost: 3 },
  'classic':       { genres: ['Drama'], keywords: ['classic','old','vintage','timeless'], boost: 1 },
  'rewatch':       { genres: [], keywords: ['rewatch','rewatchable','again','favourite','favorite','best of'], boost: 0 },
  'hidden-gem':    { genres: [], keywords: ['hidden gem','underrated','obscure','lesser known'], boost: 0 },
  'good':          { genres: ['Drama','Comedy'], keywords: ['good','great','best','top','excellent','brilliant'], boost: 1 },
}

const LANG_MAP: Record<string, string[]> = {
  'Tamil':     ['tamil','kollywood'],
  'Hindi':     ['hindi','bollywood'],
  'Malayalam': ['malayalam','mollywood'],
  'Telugu':    ['telugu','tollywood'],
  'Korean':    ['korean','korea'],
  'Japanese':  ['japanese','japan','anime'],
  'French':    ['french','france'],
  'Spanish':   ['spanish','spain'],
  'English':   ['english','hollywood'],
}

const SIMILAR_RE = /(?:similar to|like|loved?|enjoyed?|watched?|fan of)\s+["']?([A-Z][^,.!?]+?)["']?(?:\s*[,—–.!?]|$)/gi

function tokenize(q: string): string[] {
  return q.toLowerCase().replace(/[^\w\s-]/g, ' ').split(/\s+/).filter(Boolean)
}

export function needsApiCall(query: string): boolean {
  const hasSimilarRef = SIMILAR_RE.test(query)
  SIMILAR_RE.lastIndex = 0
  if (hasSimilarRef) return true
  const toks = tokenize(query)
  const allKeywords = Object.values(MOOD_MAP).flatMap(m => m.keywords)
  const allLangs = Object.values(LANG_MAP).flat()
  const hasSignal = toks.some(t => allKeywords.includes(t) || allLangs.includes(t))
  if (!hasSignal && toks.length <= 3) return true
  return false
}

export function heuristicRecommend(query: string, movies: HeuristicMovie[]): string {
  const q = query.toLowerCase()
  const toks = tokenize(q)

  // Language filter
  let langFilter: string | null = null
  for (const [lang, kws] of Object.entries(LANG_MAP)) {
    if (toks.some(t => kws.includes(t))) { langFilter = lang; break }
  }

  // Runtime filter
  let maxRuntime: number | null = null
  if (q.includes('under 2') || q.includes('under two') || q.includes('90 min')) maxRuntime = 115
  if (q.includes('under 1.5') || q.includes('under 90')) maxRuntime = 90
  if (q.includes('under 2.5') || q.includes('under 150')) maxRuntime = 150

  const wantRewatch   = toks.some(t => MOOD_MAP['rewatch'].keywords.includes(t))
  const wantHiddenGem = toks.some(t => MOOD_MAP['hidden-gem'].keywords.includes(t))

  // Build genre score map
  const moodScores = new Map<string, number>()
  for (const mood of Object.values(MOOD_MAP)) {
    const hit = toks.some(t => mood.keywords.includes(t))
    if (hit) mood.genres.forEach(g => moodScores.set(g, (moodScores.get(g) || 0) + mood.boost))
  }

  const scored: ScoredMovie[] = movies.map(m => {
    let score = 0
    const reasons: string[] = []
    const filmGenres = m.genre.split(',').map(g => g.trim())

    if (langFilter && m.language === langFilter) { score += 5; reasons.push(m.language) }
    if (langFilter && m.language !== langFilter) score -= 10

    for (const g of filmGenres) {
      const boost = moodScores.get(g) || 0
      if (boost > 0) { score += boost; reasons.push(g) }
    }

    if (m.tmdbRating >= 8.0)      score += 3
    else if (m.tmdbRating >= 7.5) score += 2
    else if (m.tmdbRating >= 7.0) score += 1
    else if (m.tmdbRating < 6.0)  score -= 2

    if (wantRewatch && m.timesWatched >= 2) { score += 4; reasons.push(`${m.timesWatched}× watched`) }
    if (m.timesWatched >= 3) score += 1
    if (wantHiddenGem && m.tmdbRating >= 7.0 && m.tmdbRating < 7.8) score += 2

    const mins = parseInt(m.runtime)
    if (maxRuntime && !isNaN(mins) && mins > maxRuntime) score -= 20

    const reason = reasons.length > 0 ? reasons.slice(0, 2).join(', ') : filmGenres[0]
    return { ...m, score, reason }
  })

  const top5 = scored
    .filter(m => m.score > 0)
    .sort((a, b) => b.score - a.score || b.tmdbRating - a.tmdbRating)
    .slice(0, 5)

  const results = top5.length > 0 ? top5 : [...movies]
    .filter(m => !langFilter || m.language === langFilter)
    .sort((a, b) => b.tmdbRating - a.tmdbRating)
    .slice(0, 5)
    .map(m => ({ ...m, score: 0, reason: m.genre.split(',')[0].trim() }))

  const lines = results.map(m => {
    const rw = m.timesWatched >= 2 ? ' ★' : ''
    return `**${m.name}** (${m.releaseYear}, ${m.language}) — ${m.genre.split(',')[0].trim()}, rated ${m.tmdbRating.toFixed(1)}${rw}.`
  })

  return lines.join('\n\n') + '\n\nWant something different? Tell me more about your mood or a film you love.'
}
