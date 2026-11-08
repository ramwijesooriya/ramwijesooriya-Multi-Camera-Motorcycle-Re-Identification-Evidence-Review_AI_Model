import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import { seekVideo, TrackVideo } from './TrackVideo';
import { Annotations } from './Annotations';
import { Matching, TimingControls, defaultTiming } from './Matching';
import type { MatchingResult, Sample, Timing } from './Matching';

type Observation = {
  frame_index: number;
  seconds: number;
  bbox: number[];
  confidence: number;
};
type Track = {
  camera_id: string;
  appearance?: {
    model: string;
    sample_count: number;
    usable_samples: number;
    samples: Sample[];
  };
  id: string;
  crop: string;
  summary: {
    start_seconds: number;
    end_seconds: number;
    observation_count: number;
    displacement_pixels: number[];
    quality_warnings: string[];
  };
  observations: Observation[];
};
type Results = {
  inputs?: Record<string, {
    sha256: string;
  }>;
  matching?: MatchingResult;
  id: string;
  elapsed_seconds: number;
  processing_fps: number;
  cameras: {
    camera_id: string;
    fps: number;
    frames_processed: number;
    tracks: Track[];
  }[];
};

async function jsonResponse(response: Response) {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(Array.isArray(body.detail) ? body.detail.map((e: {
      msg?: string;
    }) => e.msg ?? 'Invalid configuration').join('; ') : body.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function App() {
  const [timing, setTiming] = useState<Timing>(defaultTiming);
  const [files, setFiles] = useState<(File | null)[]>([null, null]);
  const [urls, setUrls] = useState<string[]>([]);
  const [job, setJob] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<Results | null>(null);
  useEffect(() => {
    const next = files.map(f => f ? URL.createObjectURL(f) : '');
    setUrls(next);
    return () => next.forEach(u => {
      if (u)
        URL.revokeObjectURL(u);
    });
  }, [files]);
  useEffect(() => {
    if (!job)
      return;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const value = await jsonResponse(await fetch(`/api/jobs/${job}`));
        if (!active)
          return;
        setStatus(value.status + (value.progress ? ` · ${value.progress.camera_id}: ${value.progress.frames_processed} frames` : ''));
        if (value.status === 'completed') {
          const data = await jsonResponse(await fetch(`/api/jobs/${job}/results`));
          if (active) {
            setResults(data);
            setBusy(false);
          }
        }
        else if (value.status === 'failed') {
          setError(value.error);
          setBusy(false);
        }
        else
          timer = setTimeout(poll, 1200);
      }
      catch (e) {
        if (active) {
          setError(String(e));
          setBusy(false);
        }
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [job]);
  function chooseFile(index: number, file: File | null) {
    setResults(null);
    setJob('');
    setStatus('');
    setError('');
    setFiles(old => old.map((f, i) => i === index ? file : f));
  }
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!files[0] || !files[1])
      return;
    setBusy(true);
    setError('');
    setResults(null);
    setJob('');
    setStatus('Uploading videos');
    const form = new FormData();
    form.append('camera_a', files[0]);
    form.append('camera_b', files[1]);
    form.append('matching_config', JSON.stringify(timing));
    try {
      const value = await jsonResponse(await fetch('/api/jobs', { method: 'POST', body: form }));
      setJob(value.id);
    }
    catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }
  return <main>
    <header>
      <p>COMPUTER VISION WORKSPACE</p>
      <h1>Motorcycle evidence review</h1>
      <p>YOLO + ByteTrack · Multi-frame visual retrieval · Independent evidence rules</p>
    </header>
    <aside>Track IDs are local to each camera. Unknown timing and ambiguous lookalikes prevent reliable acceptance. Identical-video tests do not measure cross-camera accuracy.</aside>
    <form onSubmit={submit}>
      <section className="grid">
        {['Camera A', 'Camera B'].map((label, i) => <article key={label}>
          <h2>
            {label}
          </h2>
          <input aria-label={`${label} video`} type="file" accept="video/*" disabled={busy} required onChange={e => chooseFile(i, e.target.files?.[0] ?? null)} />
          {urls[i] && <TrackVideo src={urls[i]} label={`${label} source video`} tracks={results?.cameras[i]?.tracks} fps={results?.cameras[i]?.fps} />}
        </article>)}
      </section>
      <TimingControls value={timing} setValue={setTiming} disabled={busy} />
      <button disabled={busy || !files[0] || !files[1]}>
        {busy ? 'Processing…' : 'Extract tracks and review matches'}
      </button>
    </form>
    <p role="status">
      {status}
    </p>
    {job && <p>Job: <code>
      {job}
    </code>
    </p>}
    {error && <p role="alert" className="error">
      {error}
    </p>}
    {results && <>
      <p>Processed in {results.elapsed_seconds.toFixed(1)}s · {results.processing_fps.toFixed(1)} frames/s</p>
      <a href={`/api/jobs/${job}/results`}>Download complete evidence JSON</a> · <a href={`/api/jobs/${job}/results.csv`}>Download candidate CSV</a>
      <section className="grid">
        {results.cameras.map(camera => <article key={camera.camera_id}>
          <h2>
            {camera.camera_id}
          </h2>
          <p>
            {camera.frames_processed} frames · {camera.tracks.length} tracks</p>
          {!camera.tracks.length && <p>No tracked motorcycles detected. No identity results have been inferred.</p>}
          {camera.tracks.map(track => <div className="track" key={track.id}>
            <img src={`/api/jobs/${job}/media/${track.crop}`} alt={`Evidence for ${track.id}`} />
            <div>
              <h3>
                {track.id}
              </h3>
              <p>
                {track.summary.observation_count} observations · {track.summary.start_seconds.toFixed(2)}–{track.summary.end_seconds.toFixed(2)}s</p>
              <p>Movement: {track.summary.displacement_pixels.map(v => v.toFixed(0)).join(', ')} pixels</p>
              <p>
                {track.summary.quality_warnings.join(', ')}
              </p>
              <button onClick={() => seekVideo(camera.camera_id, track.summary.start_seconds)}>Seek to track</button>
              <details>
                <summary>Frame observations</summary>
                <pre>
                  {JSON.stringify(track.observations, null, 2)}
                </pre>
              </details>
            </div>
          </div>)}
        </article>)}
      </section>
      {results.matching && <Matching matching={results.matching} tracks={results.cameras.flatMap(c => c.tracks)} job={job} onSeek={seekVideo} />}
      <Annotations key={job} job={job} tracks={results.cameras.flatMap(c => c.tracks)} inputs={results.inputs} onSeek={seekVideo} />
    </>}
  </main>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode>
  <App />
</React.StrictMode>);
