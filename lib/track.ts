export function track(event: 'page_view' | 'chat_open' | 'chat_query', path?: string) {
  fetch('/api/track', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ event, path }),
  }).catch(() => {})
}
