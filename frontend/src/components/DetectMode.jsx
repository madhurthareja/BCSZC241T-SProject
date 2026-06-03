import { useState } from 'react'
import { FileDropZone } from './FileDropZone'
import { AsciiBar }     from './AsciiBar'
import { FrameChart }   from './FrameChart'
import { detectWatermark } from '../api'

export function DetectMode() {
  const [video,     setVideo]     = useState(null)
  const [key,       setKey]       = useState('')
  const [strength,  setStrength]  = useState('medium')
  const [repeat,    setRepeat]    = useState(1)
  const [threshold, setThreshold] = useState(0.2)
  const [result,    setResult]    = useState(null)
  const [log,       setLog]       = useState([])
  const [running,   setRunning]   = useState(false)

  const print = (text, cls = '') =>
    setLog(prev => [...prev, { text, cls }])

  const run = async () => {
    if (!video || !key.trim()) return
    setLog([])
    setResult(null)
    setRunning(true)

    print('> INITIATING WATERMARK DETECTION...')
    print(`> VIDEO     : ${video.name}`)
    print(`> STRENGTH  : ${strength.toUpperCase()}`)
    print(`> THRESHOLD : ${threshold}`)
    print('')
    print('> SCANNING FRAMES...')

    try {
      const data = await detectWatermark(video, { key, strength, repeat, threshold })
      setResult(data)

      const present = data.present
      print('')
      print('> ════════════════════════════════════════', 'result-sep')
      print(`>  CONFIDENCE : ${data.confidence.toFixed(4)}`,        'result-val')
      print(`>  THRESHOLD  : ${data.threshold.toFixed(4)}`,          'result-dim')
      print(`>  FRAMES     : ${data.frame_scores.length}`,            'result-dim')
      print('')
      if (present) {
        print('>  ██  WATERMARK DETECTED  ██', 'result-ok')
      } else {
        print('>  ░░  NO WATERMARK FOUND  ░░', 'result-warn')
      }
      print('> ════════════════════════════════════════', 'result-sep')
    } catch (err) {
      print(`> ERROR: ${err.message}`, 'result-err')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div>
      <div className="panel">
        <div className="panel-title">◈ DETECT WATERMARK</div>

        <FileDropZone label="LOAD TARGET VIDEO" file={video} onFile={setVideo} />

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
            type="number" min={1} max={8}
            value={repeat} onChange={e => setRepeat(Number(e.target.value))}
            style={{ minWidth: '80px' }}
          />
        </div>

        <div className="form-row">
          <span className="form-label">THRESHOLD</span>
          <input
            type="number" min={0} max={1} step={0.01}
            value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))}
            style={{ minWidth: '100px' }}
          />
        </div>

        <button
          className="btn btn-run"
          disabled={running || !video || !key.trim()}
          onClick={run}
        >
          {running ? '[ SCANNING... ]' : '[ RUN DETECTION ]'}
        </button>
      </div>

      {log.length > 0 && (
        <div className="panel result-block">
          <div className="panel-title">◈ DETECTION REPORT</div>
          {log.map((line, i) => (
            <div key={i} className={`result-line fade-in ${line.cls}`}>{line.text}</div>
          ))}
          {running && <span className="cursor" />}

          {result && !running && (
            <>
              <div className="result-sep">── CONFIDENCE METER</div>
              <div style={{ margin: '6px 0' }}>
                <AsciiBar value={result.confidence} width={36} />
              </div>
              <FrameChart scores={result.frame_scores} threshold={result.threshold} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
