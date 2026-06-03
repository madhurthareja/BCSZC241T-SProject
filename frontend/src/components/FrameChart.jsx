import { AsciiBar } from './AsciiBar'

const MAX_ROWS = 28

export function FrameChart({ scores, threshold = 0.2 }) {
  if (!scores?.length) return null

  let display = scores
  let step    = 1

  if (scores.length > MAX_ROWS) {
    step    = Math.ceil(scores.length / MAX_ROWS)
    display = scores.filter((_, i) => i % step === 0).slice(0, MAX_ROWS)
  }

  const aboveCount = scores.filter(s => s >= threshold).length
  const pct        = ((aboveCount / scores.length) * 100).toFixed(1)

  return (
    <div style={{ marginTop: '10px' }}>
      <div className="result-sep">
        ── FRAME SCORES  [{scores.length} FRAMES  {step > 1 ? `1:${step} SAMPLE` : 'FULL'}]
        {'  '}{pct}% ABOVE THRESHOLD
      </div>

      {display.map((score, i) => {
        const frameNum = i * step
        const above    = score >= threshold
        return (
          <div
            key={i}
            className="result-line fade-in"
            style={{ color: above ? 'var(--success)' : 'var(--amber-dim)', fontSize: '13px' }}
          >
            <span style={{ color: 'var(--border-hi)', marginRight: '8px', fontSize: '12px' }}>
              {String(frameNum).padStart(4, '0')}
            </span>
            <AsciiBar value={score} width={24} />
            <span style={{ marginLeft: '8px' }}>
              {above ? '◆' : '◇'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
