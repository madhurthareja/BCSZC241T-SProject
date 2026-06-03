const MODES = [
  { key: 'embed',     label: 'EMBED',     fn: 'F1' },
  { key: 'detect',    label: 'DETECT',    fn: 'F2' },
  { key: 'calibrate', label: 'CALIBRATE', fn: 'F3' },
  { key: 'bias',      label: 'BIAS',      fn: 'F4' },
]

export function ModeBar({ mode, onMode }) {
  return (
    <div style={{
      display:       'flex',
      gap:           0,
      borderBottom:  '1px solid var(--border)',
      background:    'var(--bg)',
      flexShrink:    0,
      position:      'relative',
      zIndex:        2,
    }}>
      {MODES.map(m => {
        const active = m.key === mode
        return (
          <button
            key={m.key}
            onClick={() => onMode(m.key)}
            style={{
              background:   active ? 'var(--amber)' : 'transparent',
              color:        active ? 'var(--bg)'    : 'var(--amber-dim)',
              border:       'none',
              borderRight:  '1px solid var(--border)',
              fontFamily:   'var(--mono)',
              fontSize:     '12px',
              padding:      '8px 22px',
              cursor:       'pointer',
              letterSpacing:'2px',
              textTransform:'uppercase',
              textShadow:   active ? 'none' : undefined,
              transition:   'background 0.1s, color 0.1s',
            }}
          >
            <span style={{ color: active ? 'var(--bg)' : 'var(--border-hi)', marginRight: '6px', fontSize: '11px' }}>
              [{m.fn}]
            </span>
            {m.label}
          </button>
        )
      })}

      <div style={{
        marginLeft:  'auto',
        padding:     '8px 32px',
        fontSize:    '11px',
        color:       'var(--border-hi)',
        letterSpacing: '1px',
      }}>
        DCT SPREAD-SPECTRUM
      </div>
    </div>
  )
}
