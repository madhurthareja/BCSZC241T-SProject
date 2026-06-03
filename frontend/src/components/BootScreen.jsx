import { useEffect, useState } from 'react'

const LINES = [
  'VIDMARK TERMINAL  v1.0.0',
  '════════════════════════════════════════════',
  '',
  'BOOTING SUBSYSTEMS...',
  '  WATERMARK ENGINE    ............  OK',
  '  DCT SPREAD-SPECTRUM ............  OK',
  '  ENCODER MODULE      ............  OK',
  '  DETECTION UNIT      ............  OK',
  '  FASTAPI BACKEND     ............  LINKING',
  '',
]

export function BootScreen({ onDone }) {
  const [lines, setLines]     = useState([])
  const [bar, setBar]         = useState(0)
  const [exiting, setExiting] = useState(false)

  useEffect(() => {
    let i = 0
    const addLine = () => {
      if (i < LINES.length) {
        const line = LINES[i++]
        setLines(prev => [...prev, line])
        setTimeout(addLine, i <= 2 ? 60 : 55 + Math.random() * 55)
      } else {
        let v = 0
        const tick = setInterval(() => {
          v += Math.random() * 9 + 3
          if (v >= 100) {
            v = 100
            clearInterval(tick)
            setBar(100)
            setLines(prev => [...prev, '', '  ALL SYSTEMS  GO.', ''])
            setTimeout(() => setExiting(true), 300)
            setTimeout(onDone, 820)
          } else {
            setBar(v)
          }
        }, 35)
      }
    }
    const t = setTimeout(addLine, 180)
    return () => clearTimeout(t)
  }, [onDone])

  const filled  = Math.round((bar / 100) * 32)
  const barStr  = '█'.repeat(filled) + '░'.repeat(32 - filled)

  return (
    <div style={{
      position:       'fixed',
      inset:          0,
      background:     '#0D0D0D',
      display:        'flex',
      flexDirection:  'column',
      alignItems:     'flex-start',
      justifyContent: 'center',
      padding:        '0 10vw',
      fontFamily:     "'Share Tech Mono', monospace",
      color:          '#FFB800',
      fontSize:       '14px',
      zIndex:         9999,
      opacity:        exiting ? 0 : 1,
      transition:     'opacity 0.5s ease',
    }}>
      {lines.map((line, idx) => (
        <div
          key={idx}
          className="fade-in"
          style={{
            marginBottom:  '3px',
            textShadow:    '0 0 6px rgba(255,184,0,0.85)',
            whiteSpace:    'pre',
          }}
        >
          {line || ' '}
        </div>
      ))}

      {lines.length >= LINES.length && (
        <div style={{
          marginTop:  '12px',
          textShadow: '0 0 6px rgba(255,184,0,0.85)',
          whiteSpace: 'pre',
        }}>
          {`  LOADING  [${barStr}]  ${Math.round(bar)}%`}
        </div>
      )}
    </div>
  )
}
