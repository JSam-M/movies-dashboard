'use client'

import { useState, useRef, useEffect } from 'react'

interface Props {
  options: string[]
  selected: string[]
  onChange: (selected: string[]) => void
  placeholder?: string
  label?: string
}

export default function MultiSelect({ options, selected, onChange, placeholder = 'Search…', label }: Props) {
  const [query, setQuery]     = useState('')
  const [open,  setOpen]      = useState(false)
  const ref                   = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const filtered = options.filter(o =>
    o !== 'All' && o.toLowerCase().includes(query.toLowerCase())
  ).slice(0, 20)

  const toggle = (opt: string) => {
    if (selected.includes(opt)) onChange(selected.filter(s => s !== opt))
    else onChange([...selected, opt])
  }

  const remove = (opt: string) => onChange(selected.filter(s => s !== opt))

  const inputStyle: React.CSSProperties = {
    width: '100%', padding: '7px 10px',
    borderRadius: '10px', border: '1px solid rgba(0,0,0,0.08)',
    background: 'rgba(0,0,0,0.04)', fontFamily: 'inherit',
    fontSize: '0.78rem', color: 'var(--text)', outline: 'none',
  }

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      {label && (
        <div style={{
          fontSize: '0.58rem', fontWeight: 600, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: 'var(--sub)', marginBottom: '6px',
          fontFamily: 'inherit',
        }}>{label}</div>
      )}

      {/* Selected tags */}
      {selected.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginBottom: '6px' }}>
          {selected.map(s => (
            <span key={s} style={{
              display: 'inline-flex', alignItems: 'center', gap: '4px',
              padding: '3px 8px', borderRadius: '100px',
              background: 'rgba(0,113,227,0.1)', color: '#0071e3',
              fontSize: '0.72rem', fontFamily: 'inherit',
            }}>
              {s}
              <button onClick={() => remove(s)} style={{
                background: 'none', border: 'none', cursor: 'pointer',
                color: '#0071e3', fontSize: '13px', lineHeight: 1,
                padding: 0, opacity: 0.7,
              }}>×</button>
            </span>
          ))}
        </div>
      )}

      {/* Search input */}
      <input
        style={inputStyle}
        placeholder={selected.length > 0 ? 'Add more…' : placeholder}
        value={query}
        onChange={e => { setQuery(e.target.value); setOpen(true) }}
        onFocus={() => setOpen(true)}
      />

      {/* Dropdown */}
      {open && (filtered.length > 0 || query.length === 0) && options.filter(o => o !== 'All').slice(0,20).length > 0 && (
        <div style={{
          position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
          background: 'white', border: '1px solid rgba(0,0,0,0.1)',
          borderRadius: '12px', boxShadow: '0 8px 24px rgba(0,0,0,0.1)',
          zIndex: 100, overflow: 'hidden', maxHeight: '220px', overflowY: 'auto',
        }}>
          {(query.length > 0 ? filtered : options.filter(o => o !== 'All').slice(0,20)).map(opt => {
            const isSelected = selected.includes(opt)
            return (
              <div key={opt} onClick={() => { toggle(opt); setQuery('') }}
                style={{
                  padding: '9px 12px', fontSize: '0.8rem', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  background: isSelected ? 'rgba(0,113,227,0.04)' : 'white',
                  color: isSelected ? '#0071e3' : 'var(--text)',
                  fontFamily: 'inherit',
                }}
                onMouseEnter={e => (e.currentTarget.style.background = isSelected ? 'rgba(0,113,227,0.08)' : 'rgba(0,0,0,0.03)')}
                onMouseLeave={e => (e.currentTarget.style.background = isSelected ? 'rgba(0,113,227,0.04)' : 'white')}
              >
                {opt}
                {isSelected && (
                  <div style={{
                    width: '16px', height: '16px', borderRadius: '50%',
                    background: '#0071e3', display: 'flex', alignItems: 'center',
                    justifyContent: 'center', flexShrink: 0,
                  }}>
                    <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
