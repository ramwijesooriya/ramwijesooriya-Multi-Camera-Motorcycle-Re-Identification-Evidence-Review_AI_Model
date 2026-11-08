import { useEffect, useState } from 'react';

type AnnotationTrack = {
  id: string;
  camera_id: string;
  summary: {
    start_seconds: number;
    end_seconds: number;
  };
};
type Label = {
  track_id: string;
  camera_id: string;
  label_status: 'unknown' | 'confirmed' | 'excluded_mixed';
  identity_id: string;
  query_status: 'unknown' | 'has_counterpart' | 'no_counterpart';
  fragment_group: string;
  notes: string;
};
type Annotations = {
  schema_version: number;
  job_id: string;
  source_input_hashes: Record<string, string>;
  footage_relationship: 'unconfirmed' | 'independent_cameras' | 'same_scene_duplicate';
  annotation_basis: string;
  tracks: Label[];
};

export function makeAnnotationTemplate(job: string, tracks: AnnotationTrack[], inputs: Record<string, {
  sha256: string;
}> = {}): Annotations {
  return {
    schema_version: 1,
    job_id: job,
    source_input_hashes: Object.fromEntries(Object.entries(inputs).map(([camera, value]) => [camera, value.sha256])),
    footage_relationship: 'unconfirmed',
    annotation_basis: '',
    tracks: tracks.map(t => ({
      track_id: t.id,
      camera_id: t.camera_id,
      label_status: 'unknown',
      identity_id: '',
      query_status: 'unknown',
      fragment_group: '',
      notes: ''
    }))
  };
}

export function Annotations({ job, tracks, inputs = {}, onSeek }: {
  job: string;
  tracks: AnnotationTrack[];
  inputs?: Record<string, {
    sha256: string;
  }>;
  onSeek: (camera: string, seconds: number) => void;
}) {
  const storageKey = `motorcycle-ground-truth:${job}`;
  const [labels, setLabels] = useState<Annotations>(() => {
    const initial = makeAnnotationTemplate(job, tracks, inputs);
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
      if (saved?.job_id === job && Array.isArray(saved.tracks) && saved.tracks.map((t: Label) => t.track_id).sort().join('|') === tracks.map(t => t.id).sort().join('|'))
        return saved;
    }
    catch { /* Storage may be unavailable; no labels are guessed. */ }
    return initial;
  });
  const [storageError, setStorageError] = useState('');
  useEffect(() => {
    try {
      localStorage.setItem(storageKey, JSON.stringify(labels));
      setStorageError('');
    }
    catch {
      setStorageError('Browser draft storage is unavailable. Download the labels JSON to retain your annotations.');
    }
  }, [labels, storageKey]);
  function change(index: number, patch: Partial<Label>) {
    setLabels(old => ({ ...old, tracks: old.tracks.map((label, i) => i === index ? { ...label, ...patch } : label) }));
  }
  function download() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(labels, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = `${job}-labels.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  return <section className="matching">
    <h2>Manual ground-truth annotation</h2>
    <aside>These are human labels for offline evaluation, not model decisions. They never change candidate scores or force acceptance. Do not label from similarity alone. Use the same identity name across cameras only when independently confirmed.</aside>
    <label>Relationship between source videos<select value={labels.footage_relationship} onChange={e => setLabels({ ...labels, footage_relationship: e.target.value as Annotations['footage_relationship'] })}>
      <option value="unconfirmed">Not yet established</option>
      <option value="independent_cameras">Genuinely different cameras (manually verified)</option>
      <option value="same_scene_duplicate">Same scene / duplicate footage (not valid evaluation)</option>
    </select>
    </label>
    <label>Independent annotation evidence and provenance<textarea value={labels.annotation_basis} onChange={e => setLabels({ ...labels, annotation_basis: e.target.value })} placeholder="Describe the original footage, continuity or independent records establishing the identities. Do not use model scores as ground truth." />
    </label>
    {labels.tracks.map((label, index) => {
      const track = tracks.find(t => t.id === label.track_id)!;
      return <article className="annotation-row" key={label.track_id}>
        <h3>
          {label.track_id} · {track.summary.start_seconds.toFixed(2)}–{track.summary.end_seconds.toFixed(2)}s</h3>
        <button onClick={() => onSeek(label.camera_id, track.summary.start_seconds)}>Inspect source track</button>
        <div className="config-grid">
          <label>Identity evidence status<select value={label.label_status} onChange={e => change(index, { label_status: e.target.value as Label['label_status'] })}>
            <option value="unknown">Unknown — do not infer a negative</option>
            <option value="confirmed">Confirmed manually</option>
            <option value="excluded_mixed">Exclude mixed / switched identity track</option>
          </select>
          </label>
          <label>Physical identity label<input value={label.identity_id} onChange={e => change(index, { identity_id: e.target.value })} placeholder="Assign from independent evidence" />
          </label>
          {label.camera_id === 'camera_a' && <label>Camera B counterpart status<select value={label.query_status} onChange={e => change(index, { query_status: e.target.value as Label['query_status'] })}>
            <option value="unknown">Unknown</option>
            <option value="has_counterpart">Known to have a counterpart</option>
            <option value="no_counterpart">Confirmed no counterpart in B clip</option>
          </select>
          </label>}
          <label>Optional same-camera fragment group<input value={label.fragment_group} onChange={e => change(index, { fragment_group: e.target.value })} placeholder="Only if continuity is manually established" />
          </label>
          <label>Notes<input value={label.notes} onChange={e => change(index, { notes: e.target.value })} />
          </label>
        </div>
      </article>;
    })}
    <p>Drafts are stored only in this browser. Download JSON for the evaluation script. Fragment groups are human annotations, not automatic tracker merges; repeated tracks may be a repeat passage rather than fragmentation.</p>
    {storageError && <p role="alert">
      {storageError}
    </p>}
    <button onClick={download}>Download ground-truth labels JSON</button>
  </section>;
}
