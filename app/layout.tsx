import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Film Collection',
  description: 'A personal film archive and AI recommendation engine',
  icons: {
    icon: [
      { url: '/favicon.svg', type: 'image/svg+xml' },
      { url: '/favicon.png', type: 'image/png', sizes: '32x32' },
    ],
    apple: '/apple-touch-icon.png',
  },
}
export const viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
}
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" style={{touchAction: 'manipulation'}}>
      <body>{children}</body>
    </html>
  )
}
