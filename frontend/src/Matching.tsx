import { useState } from 'react';

export type Sample = {
  crop: string;
  seconds: number;
  frame_index: number;
  usable: boolean;
  blur_variance: number;
};
export type Evidence = {
  kind: 'supporting' | 'contradictory' | 'missing';
  code: string;
  message: string;
  value: unknown;
  blocking: boolean;
};
export type Candidate = {
  query_track_id: string;
  candidate_track_id: string;
  rank: number;
  similarity: number;
  competition: {
    margin: number | null;
    reverse_margin: number | null;
    ambiguous: boolean;
    near_tie?: boolean;
    outcompeted?: boolean;
    reverse_best_query_track_id?: string;
    reverse_best_similarity?: number;
    mutual_best: boolean;
    competing_candidates: {
      track_id: string;
      similarity: number;
    }[];
    competing_queries: {
      track_id: string;
      similarity: number;
    }[];
  };
  verification: {
    outcome: 'ACCEPTED' | 'UNCERTAIN' | 'REJECTED';
    independent_visual_score: number | null;
    evidence: Evidence[];
    uncertainty: {
      blocking_reasons: string[];
    };
  };
};
export type MatchingResult = {
  fragmentation_hypotheses?: unknown[];
  config?: Record<string, unknown>;
  encoder?: Record<string, unknown>;
  status: string;
  error?: string;
  candidates: Candidate[];
  queries: {
    query_track_id: string;
    result: string;
  }[];
  evaluation?: {
    identical_inputs: boolean;
    note: string;
  };
  model_limitations?: string[];
};
type TrackView = {
  id: string;
  camera_id?: string;
  crop: string;
  appearance?: {
    model: string;
    sample_count: number;
    usable_samples: number;
    samples: Sample[];
  };
};
export type Timing = {
  temporal_enabled: boolean;
  camera_b_offset_seconds: number | null;
  clock_uncertainty_seconds: number;
  boundary_uncertainty_seconds: number;
  min_travel_seconds: number;
  max_travel_seconds: number;
  route: 'a_to_b' | 'b_to_a';
  camera_a_direction: string;
  camera_b_direction: string;
};
export const defaultTiming: Timing = {
  temporal_enabled: true,
  camera_b_offset_seconds: null,
  clock_uncertainty_seconds: 0,
  boundary_uncertainty_seconds: 1,
  min_travel_seconds: 0,
  max_travel_seconds: 120,
  route: 'a_to_b',
  camera_a_direction: 'unknown',
  camera_b_direction: 'unknown'
};

export function TimingControls({ value, setValue, disabled }: {
  value: Timing;
  setValue: (v: Timing) => void;
  disabled: boolean;
}) {
  const number = (key: 'clock_uncertainty_seconds' | 'boundary_uncertainty_seconds' | 'min_travel_seconds' | 'max_travel_seconds', label: string) => <label key={key}>
    {label}
    <input type="number" step="any" min="0" value={value[key]} required onChange={e => setValue({ ...value, [key]: Number(e.target.value) })} />
  </label>;
  return <fieldset disabled={disabled}>
    <legend>Camera timing and route evidence</legend>
    <label>
      <input type="checkbox" checked={value.temporal_enabled} onChange={e => setValue({ ...value, temporal_enabled: e.target.checked })} /> Check travel timing</label>
    <p>Add the signed offset to Camera B timestamps to align them with Camera A. Include recording-start differences and clock correction. Leave unknown if unavailable; missing timing prevents acceptance.</p>
    <div className="config-grid">
      <label>B → A offset (seconds; blank = unknown)<input type="number" step="any" value={value.camera_b_offset_seconds ?? ''} onChange={e => setValue({ ...value, camera_b_offset_seconds: e.target.value === '' ? null : Number(e.target.value) })} />
      </label>
      {number('clock_uncertainty_seconds', 'Clock/offset uncertainty ± seconds')}
      {number('boundary_uncertainty_seconds', 'Entry/exit observation uncertainty ± seconds')}
      {number('min_travel_seconds', 'Minimum route travel seconds')}
      {number('max_travel_seconds', 'Maximum route travel seconds')}
      <label>Route<select value={value.route} onChange={e => setValue({ ...value, route: e.target.value as Timing['route'] })}>
        <option value="a_to_b">A → B</option>
        <option value="b_to_a">B → A</option>
      </select>
      </label>
      {(['camera_a_direction', 'camera_b_direction'] as const).map(key => <label key={key}>
        {key === 'camera_a_direction' ? 'Camera A' : 'Camera B'} expected image motion<select value={value[key]} onChange={e => setValue({ ...value, [key]: e.target.value })}>
          {['unknown', 'left', 'right', 'up', 'down'].map(x => <option key={x}>
            {x}
          </option>)}
        </select>
      </label>)}
    </div>
    <p>Directions apply to each camera independently. Pixel motion is not world-space direction. Thresholds are experimental; no accuracy calibration has been performed. Changing these controls applies to the next job.</p>
  </fieldset>;
}

export function Matching({ matching, tracks, job, onSeek }: {
  matching: MatchingResult;
  tracks: TrackView[];
  job: string;
  onSeek: (camera: string, seconds: number) => void;
}) {
  const [selected, setSelected] = useState('');
  if (matching.status !== 'completed')
    return <aside role="alert">
      {matching.error ?? 'Matching is unavailable for this job. Track extraction remains available.'}
    </aside>;
  const query = matching.queries.find(q => q.query_track_id === selected) ?? matching.queries[0];
  const byId = Object.fromEntries(tracks.map(t => [t.id, t]));
  const cards = matching.candidates.filter(c => c.query_track_id === query?.query_track_id);
  function gallery(id: string) {
    const t = byId[id];
    if (!t)
      return <p>Track evidence unavailable.</p>;
    return <div>
      <h3>
        {id}
      </h3>
      <img className="comparison-image" src={`/api/jobs/${job}/media/${t.crop}`} alt={`Best crop ${id}`} />
      <p>
        {t.appearance?.usable_samples ?? 0} usable / {t.appearance?.sample_count ?? 0} sampled frames</p>
      <div className="crop-strip">
        {t.appearance?.samples.map(s => <button key={s.frame_index} title={`${s.seconds.toFixed(2)}s · blur variance ${s.blur_variance.toFixed(1)}`} onClick={() => onSeek(t.camera_id ?? id.split(':')[0], s.seconds)}>
          <img src={`/api/jobs/${job}/media/${s.crop}`} alt={`${id} frame ${s.frame_index}`} />
          <span>
            {s.seconds.toFixed(2)}s{s.usable ? '' : ' · poor quality'}
          </span>
        </button>)}
      </div>
    </div>;
  }
  return <section className="matching">
    <h2>Cross-camera candidates and evidence review</h2>
    <aside>Cosine scores measure descriptor similarity, not identity probability. ACCEPTED is an uncalibrated rule outcome. A highest-ranked candidate may remain uncertain or be rejected.</aside>
    {matching.evaluation?.identical_inputs && <p className="error" role="alert">Identical input videos detected. This is a smoke test, not external-camera evaluation; acceptance is blocked.</p>}
    <p>
      {matching.evaluation?.note}
    </p>
    {query ? <>
      <label>Query motorcycle track <select value={query.query_track_id} onChange={e => setSelected(e.target.value)}>
        {matching.queries.map(q => <option key={q.query_track_id} value={q.query_track_id}>
          {q.query_track_id} · {q.result.replaceAll('_', ' ')}
        </option>)}
      </select>
      </label>
      <h3>
        {query.result.replaceAll('_', ' ')}
      </h3>
      {!cards.length && <p>No comparable destination track has a usable descriptor.</p>}
      {cards.map(c => <article className="candidate" key={c.candidate_track_id}>
        <div className="candidate-title">
          <h3>#{c.rank} · {c.query_track_id} ↔ {c.candidate_track_id}
          </h3>
          <strong className={`badge ${c.verification.outcome.toLowerCase()}`}>
            {c.verification.outcome}
          </strong>
        </div>
        <p>Track cosine: <b>
          {c.similarity.toFixed(4)}
        </b> · Independent crop agreement: {c.verification.independent_visual_score?.toFixed(4) ?? 'unavailable'} · Margin: {c.competition.margin?.toFixed(4) ?? 'no alternative'} · Reciprocal margin: {c.competition.reverse_margin?.toFixed(4) ?? 'no alternative'}
        </p>
        {c.competition.outcompeted ? <p className="warning">This pair is outscored by another track. Camera B prefers {c.competition.reverse_best_query_track_id ?? 'another query'} at cosine {c.competition.reverse_best_similarity?.toFixed(4) ?? 'unavailable'}. A negative margin does not prove different physical identity; inspect possible track fragmentation.</p> : c.competition.ambiguous && <p className="warning">Visually similar competing tracks: identity is ambiguous.</p>}
        <div className="grid">
          {gallery(c.query_track_id)}
          {gallery(c.candidate_track_id)}
        </div>
        <details>
          <summary>Competing matches and queries</summary>
          <pre>
            {JSON.stringify(c.competition, null, 2)}
          </pre>
        </details>
        <div className="evidence-grid">
          {(['supporting', 'contradictory', 'missing'] as const).map(kind => <div key={kind}>
            <h4>
              {kind === 'missing' ? 'Missing / uncertain' : kind}
            </h4>
            {c.verification.evidence.filter(e => e.kind === kind).map((e, i) => <div className="evidence" key={`${e.code}-${i}`}>
              <b>
                {e.code.replaceAll('_', ' ')}
              </b>
              {e.blocking && <span> · blocks acceptance</span>}
              <p>
                {e.message}
              </p>
              {e.value !== null && <details>
                <summary>Measured evidence</summary>
                <pre>
                  {JSON.stringify(e.value, null, 2)}
                </pre>
              </details>}
            </div>)}
            {!c.verification.evidence.some(e => e.kind === kind) && <p>None recorded.</p>}
          </div>)}
        </div>
      </article>)}
    </> : <p>No query motorcycle tracks were extracted.</p>}
    <details>
      <summary>Possible camera-local track fragments (unconfirmed; no automatic merging)</summary>
      <pre>
        {JSON.stringify(matching.fragmentation_hypotheses ?? [], null, 2)}
      </pre>
    </details>
    <details>
      <summary>Saved job configuration and encoder provenance</summary>
      <pre>
        {JSON.stringify({ config: matching.config, encoder: matching.encoder }, null, 2)}
      </pre>
    </details>
    <details>
      <summary>Model and review limitations</summary>
      <ul>
        {matching.model_limitations?.map(text => <li key={text}>
          {text}
        </li>)}
      </ul>
    </details>
  </section>;
}
