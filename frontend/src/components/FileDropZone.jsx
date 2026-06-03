import { useRef, useState } from 'react'

function fmtSize(bytes) {
  if (bytes < 1024)        return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function FileDropZone({ label = 'LOAD VIDEO', file, onFile, accept = 'video/*' }) {
  const inputRef     = useRef(null)
  const [drag, setDrag] = useState(false)

  const handle = (f) => { if (f) onFile(f) }

  return (
    <div
      className={`file-drop${drag ? ' dragover' : ''}${file ? ' has-file' : ''}`}
      onClick={() => inputRef.current.click()}
      onDragOver={e  => { e.preventDefault(); setDrag(true)  }}
      onDragLeave={() => setDrag(false)}
      onDrop={e => {
        e.preventDefault()
        setDrag(false)
        handle(e.dataTransfer.files[0])
      }}
    >
      <span className="file-drop-icon">
        {file ? '▶' : '□'}
      </span>

      <span>
        {file
          ? <><span className="glow-sm">{file.name}</span>{'  '}<span style={{ color: 'var(--amber-dim)', fontSize: '12px' }}>[{fmtSize(file.size)}]</span></>
          : <span style={{ letterSpacing: '1px' }}>&gt; {label}: DROP FILE OR CLICK</span>
        }
      </span>

      {file && (
        <button
          onClick={e => { e.stopPropagation(); onFile(null) }}
          style={{
            marginLeft:  'auto',
            background:  'transparent',
            border:      '1px solid var(--border-hi)',
            color:       'var(--amber-dim)',
            fontFamily:  'var(--mono)',
            fontSize:    '11px',
            padding:     '2px 8px',
            cursor:      'pointer',
            letterSpacing: '1px',
          }}
        >
          CLR
        </button>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={e => handle(e.target.files[0])}
      />
    </div>
  )
}
