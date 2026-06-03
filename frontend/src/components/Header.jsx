import { useEffect, useState } from 'react'

function Clock() {
  const [t, setT] = useState(new Date())
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000)
    return () => clearInterval(id)
  }, [])
  return (
    <span className="glow-sm" style={{ color: 'var(--amber-dim)' }}>
      {t.toLocaleTimeString('en-GB', { hour12: false })}
    </span>
  )
}

export function Header({ status }) {
  return (
    <div style={{
      borderBottom:  '1px solid var(--border)',
      padding:       '8px 32px',
      display:       'flex',
      alignItems:    'center',
      gap:           '16px',
      flexShrink:    0,
      background:    'var(--panel)',
      position:      'relative',
      zIndex:        2,
    }}>
      <span className="glow" style={{ fontSize: '18px', letterSpacing: '6px', fontWeight: 'bold' }}>
        VIDMARK
      </span>

      <span style={{ color: 'var(--border-hi)', fontSize: '12px' }}>
        ══════
      </span>

      <span style={{ fontSize: '11px', color: 'var(--amber-dim)', letterSpacing: '2px' }}>
        VIDEO WATERMARKING TERMINAL
      </span>

      <span style={{ marginLeft: 'auto', display: 'flex', gap: '20px', alignItems: 'center', fontSize: '12px' }}>
        <span style={{ color: status === 'online' ? 'var(--success)' : 'var(--danger)', letterSpacing: '1px' }}
              className={status === 'online' ? 'glow-success' : 'glow-danger'}>
          ◆ {status === 'online' ? 'BACKEND ONLINE' : status === 'checking' ? 'CONNECTING...' : 'BACKEND OFFLINE'}
        </span>
        <Clock />
      </span>
    </div>
  )
}
