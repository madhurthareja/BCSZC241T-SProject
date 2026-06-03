export function AsciiBar({ value, max = 1, width = 20, showValue = true }) {
  const ratio  = Math.max(0, Math.min(1, value / max))
  const filled = Math.round(ratio * width)
  const bar    = '█'.repeat(filled) + '░'.repeat(width - filled)
  return (
    <span className="ascii-bar">
      [{bar}]{showValue ? ` ${value.toFixed(3)}` : ''}
    </span>
  )
}
