import { useState } from 'react'
import { FileDropZone } from './FileDropZone'
import { embedWatermark } from '../api'

export function EmbedMode() {
  const [video,    setVideo]    = useState(null)
  const [key,      setKey]      = useState('')
  const [strength, setStrength] = useState('medium')
  const [repeat,   setRepeat]   = useState(1)
  const [log,      setLog]      = useState([])
  const [running,  setRunning]  = useState(false)

  const print = (text, cls = '') =>
    setLog(prev => [...prev, { text, cls }])

  const run = async () => {
    if (!video || !key.trim()) return
    setLog([])
    setRunning(true)

    print('> INITIATING WATERMARK EMBED SEQUENCE...')
    print(`> VIDEO    : ${video.name}`)
    print(`> KEY      : ${'*'.repeat(key.length)} (${key.length} chars)`)
    print(`> STRENGTH : ${strength.toUpperCase()}`)
    print(`> REPEAT   : ${repeat}`)
    print('')
    print('> PROCESSING FRAMES  ...  PLEASE WAIT')

    try {
      const blob   = await embedWatermark(video, { key, strength, repeat })
      const url    = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href     = url
      anchor.download = 'watermarked_video.mp4'

      print('')
      print('> ════════════════════════════════════════', 'result-sep')
      print('>  STATUS   : WATERMARK EMBEDDED SUCCESSFULLY', 'result-ok')
      print('>  OUTPUT   : watermarked_video.mp4', 'result-val')
      print('> ════════════════════════════════════════', 'result-sep')
      print('')
      print('> DOWNLOAD READY — INITIATING TRANSFER...')

      anchor.click()
      URL.revokeObjectURL(url)
    } catch (err) {
      print('')
      print(`> ERROR: ${err.message}`, 'result-err')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div>
      <div className="panel">
        <div className="panel-title">◈ EMBED WATERMARK</div>

        <FileDropZone label="LOAD SOURCE VIDEO" file={video} onFile={setVideo} />

        <div className="divider" />

        <div className="form-row">
          <span className="form-label">KEY</span>
          <input
            type="password"
            placeholder="enter watermark key..."
            value={key}
            onChange={e => setKey(e.target.value)}
          />
        </div>

        <div className="form-row">
          <span className="form-label">STRENGTH</span>
          <select value={strength} onChange={e => setStrength(e.target.value)}>
            <option value="low">LOW</option>
            <option value="medium">MEDIUM</option>
            <option value="high">HIGH</option>
          </select>
        </div>

        <div className="form-row">
          <span className="form-label">REPEAT</span>
          <input
            type="number"
            min={1} max={8}
            value={repeat}
            onChange={e => setRepeat(Number(e.target.value))}
            style={{ minWidth: '80px' }}
          />
        </div>

        <button
          className="btn btn-run"
          disabled={running || !video || !key.trim()}
          onClick={run}
        >
          {running ? '[ EMBEDDING... ]' : '[ EMBED WATERMARK ]'}
        </button>
      </div>

      {log.length > 0 && (
        <div className="panel result-block">
          <div className="panel-title">◈ OUTPUT LOG</div>
          {log.map((line, i) => (
            <div key={i} className={`result-line fade-in ${line.cls}`}>
              {line.text}
            </div>
          ))}
          {running && <span className="cursor" />}
        </div>
      )}
    </div>
  )
}
