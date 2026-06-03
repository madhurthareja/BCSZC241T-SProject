import { useState } from 'react'
import { FileDropZone } from './FileDropZone'
import { AsciiBar }     from './AsciiBar'
import { analyzeBias }  from '../api'

function GapIndicator({ gap }) {
  const abs     = Math.abs(gap)
  const healthy = abs >= 0.1
  const label   = healthy ? 'GOOD SEPARATION' : 'WEAK SEPARATION'
  const cls     = healthy ? 'result-ok' : 'result-warn'
  return (
    <div style={{ marginTop: '10px' }}>
      <div style={{ color: 'var(--amber-dim)', fontSize: '12px', marginBottom: '6px' }}>
        BIAS GAP
      </div>
      <div className={cls} style={{ fontSize: '22px', letterSpacing: '4px', fontFamily: 'var(--mono)' }}>
        {gap >= 0 ? '+' : ''}{gap.toFixed(4)}
      </div>
      <div className={`${cls} result-line`} style={{ marginTop: '6px', fontSize: '13px' }}>
        ▶  {label}
        {'  '}
        <AsciiBar value={abs} width={20} showValue={false} />
        {'  '}{(abs * 100).toFixed(1)}%
      </div>
    </div>
  )
}

export function BiasMode() {
  const [cleanFile, setCleanFile] = useState(null)
  const [wmFile,    setWmFile]    = useState(null)
  const [key,       setKey]       = useState('')
  const [strength,  setStrength]  = useState('medium')
  const [repeat,    setRepeat]    = useState(1)
  const [result,    setResult]    = useState(null)
  const [log,       setLog]       = useState([])
  const [running,   setRunning]   = useState(false)

  const print = (text, cls = '') =>
    setLog(prev => [...prev, { text, cls }])

  const run = async () => {
    if (!cleanFile || !wmFile || !key.trim()) return
    setLog([])
    setResult(null)
    setRunning(true)

    print('> INITIATING BIAS ANALYSIS...')
    print(`> CLEAN VIDEO      : ${cleanFile.name}`)
    print(`> WATERMARKED VIDEO: ${wmFile.name}`)
    print(`> STRENGTH         : ${strength.toUpperCase()}`)
    print('')
    print('> COMPUTING CONFIDENCE DELTA...')

    try {
      const data = await analyzeBias(cleanFile, wmFile, { key, strength, repeat })
      setResult(data)

      print('')
      print('> ════════════════════════════════════════', 'result-sep')
      print(`>  CLEAN CONFIDENCE      : ${data.clean_confidence.toFixed(4)}`,      'result-dim')
      print(`>  WATERMARKED CONFIDENCE: ${data.watermarked_confidence.toFixed(4)}`, 'result-val')
      print(`>  BIAS GAP              : ${data.bias_gap >= 0 ? '+' : ''}${data.bias_gap.toFixed(4)}`,
        data.bias_gap >= 0.1 ? 'result-ok' : 'result-warn')
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
        <div className="panel-title">◈ BIAS ANALYSIS</div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '12px' }}>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--amber-dim)', marginBottom: '6px', letterSpacing: '1px' }}>
              CHANNEL A — CLEAN
            </div>
            <FileDropZone label="CLEAN VIDEO" file={cleanFile} onFile={setCleanFile} />
          </div>
          <div>
            <div style={{ fontSize: '11px', color: 'var(--amber-dim)', marginBottom: '6px', letterSpacing: '1px' }}>
              CHANNEL B — WATERMARKED
            </div>
            <FileDropZone label="WATERMARKED VIDEO" file={wmFile} onFile={setWmFile} />
          </div>
        </div>

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

        <button
          className="btn btn-run"
          disabled={running || !cleanFile || !wmFile || !key.trim()}
          onClick={run}
        >
          {running ? '[ ANALYZING... ]' : '[ RUN BIAS ANALYSIS ]'}
        </button>
      </div>

      {log.length > 0 && (
        <div className="panel result-block">
          <div className="panel-title">◈ BIAS REPORT</div>
          {log.map((line, i) => (
            <div key={i} className={`result-line fade-in ${line.cls}`}>{line.text}</div>
          ))}
          {running && <span className="cursor" />}

          {result && !running && (
            <>
              <div className="result-sep">── CONFIDENCE COMPARISON</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', marginTop: '8px' }}>
                <div>
                  <span style={{ color: 'var(--amber-dim)', fontSize: '12px', marginRight: '8px' }}>
                    CLEAN
                  </span>
                  <AsciiBar value={result.clean_confidence} width={30} />
                </div>
                <div>
                  <span style={{ color: 'var(--success)', fontSize: '12px', marginRight: '8px' }}>
                    WATERMARKED
                  </span>
                  <AsciiBar value={result.watermarked_confidence} width={30} />
                </div>
              </div>
              <GapIndicator gap={result.bias_gap} />
            </>
          )}
        </div>
      )}
    </div>
  )
}
