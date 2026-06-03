const BASE = '/api'

async function post(path, form) {
  const res = await fetch(`${BASE}${path}`, { method: 'POST', body: form })
  if (!res.ok) {
    const msg = await res.text().catch(() => res.statusText)
    throw new Error(`${res.status} — ${msg}`)
  }
  return res
}

export async function checkHealth() {
  const res = await fetch(`${BASE}/health`)
  return res.ok
}

export async function embedWatermark(videoFile, { key, strength, repeat }) {
  const form = new FormData()
  form.append('video',    videoFile)
  form.append('key',      key)
  form.append('strength', strength)
  form.append('repeat',   String(repeat))
  const res = await post('/watermark/embed', form)
  return res.blob()
}

export async function detectWatermark(videoFile, { key, strength, repeat, threshold }) {
  const form = new FormData()
  form.append('video',     videoFile)
  form.append('key',       key)
  form.append('strength',  strength)
  form.append('repeat',    String(repeat))
  form.append('threshold', String(threshold))
  const res = await post('/watermark/detect', form)
  return res.json()
}

export async function calibrateThreshold(cleanFile, wmFile, { key, strength, repeat }) {
  const form = new FormData()
  form.append('clean_video',       cleanFile)
  form.append('watermarked_video', wmFile)
  form.append('key',               key)
  form.append('strength',          strength)
  form.append('repeat',            String(repeat))
  const res = await post('/watermark/calibrate', form)
  return res.json()
}

export async function analyzeBias(cleanFile, wmFile, { key, strength, repeat }) {
  const form = new FormData()
  form.append('clean_video',       cleanFile)
  form.append('watermarked_video', wmFile)
  form.append('key',               key)
  form.append('strength',          strength)
  form.append('repeat',            String(repeat))
  const res = await post('/watermark/analyze-bias', form)
  return res.json()
}
