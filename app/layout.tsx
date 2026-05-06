import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Film Collection',
  description: 'A personal film archive and AI recommendation engine',
  icons: { icon: '/favicon.svg' },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
