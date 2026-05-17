function getVisitorId(): string {
  try {
    const key = 'fc_visitor_id'
    let id = localStorage.getItem(key)
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem(key, id)
    }
    return id
  } catch {
    return 'anonymous'
  }
}

export function track(event: 'page_view' | 'chat_open' | 'chat_query', path?: string) {
  const visitorId = getVisitorId()
  const referrer = (() => {
    try {
      return document.referrer ? new URL(document.referrer).hostname : ''
    } catch { return '' }
  })()
  fetch('/api/track', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event, path, visitorId, referrer }),
  }).catch(() => {})
}
