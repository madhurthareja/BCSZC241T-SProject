import { useEffect, useState } from 'react'
import { BootScreen }    from './components/BootScreen'
import { Header }        from './components/Header'
import { ModeBar }       from './components/ModeBar'
import { EmbedMode }     from './components/EmbedMode'
import { DetectMode }    from './components/DetectMode'
import { CalibrateMode } from './components/CalibrateMode'
import { BiasMode }      from './components/BiasMode'
import { checkHealth }   from './api'
import './styles/main.css'

export default function App() {
  const [booted, setBooted] = useState(false)
  const [mode,   setMode]   = useState('embed')
  const [status, setStatus] = useState('checking')

  useEffect(() => {
    if (!booted) return
    checkHealth()
      .then(ok => setStatus(ok ? 'online' : 'offline'))
      .catch(() => setStatus('offline'))
  }, [booted])

  return (
    <>
      {!booted && <BootScreen onDone={() => setBooted(true)} />}

      {booted && (
        <div className="crt-screen">
          <Header status={status} />
          <ModeBar mode={mode} onMode={setMode} />
          <div className="content-area">
            {mode === 'embed'     && <EmbedMode />}
            {mode === 'detect'    && <DetectMode />}
            {mode === 'calibrate' && <CalibrateMode />}
            {mode === 'bias'      && <BiasMode />}
          </div>
        </div>
      )}
    </>
  )
}
