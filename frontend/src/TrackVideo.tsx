import { useEffect, useRef, useState } from 'react';

export type BoxObservation = {
  frame_index: number;
  seconds: number;
  bbox: number[];
  confidence: number;
};
export type OverlayTrack = {
  id: string;
  observations: BoxObservation[];
};

export function visibleObservations(tracks: OverlayTrack[], seconds: number, fps: number) {
  const tolerance = 1.5 / (Number.isFinite(fps) && fps > 0 ? fps : 30);
  return tracks.flatMap(track => {
    // Binary search avoids rescanning an entire video history every display frame.
    const observations = track.observations;
    let low = 0, high = observations.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (observations[middle].seconds < seconds)
        low = middle + 1;
      else
        high = middle;
    }
    const nearest = [observations[low - 1], observations[low]].filter((o): o is BoxObservation => !!o).sort((a, b) => Math.abs(a.seconds - seconds) - Math.abs(b.seconds - seconds))[0];
    return nearest && Math.abs(nearest.seconds - seconds) <= tolerance ? [{ trackId: track.id, observation: nearest }] : [];
  });
}

export function seekVideo(camera: string, seconds: number): void {
  const video = document.querySelectorAll('video')[camera === 'camera_a' ? 0 : 1];
  if (video)
    video.currentTime = seconds;
}

export function TrackVideo({ src, label, tracks = [], fps = 30 }: {
  src: string;
  label: string;
  tracks?: OverlayTrack[];
  fps?: number;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const [time, setTime] = useState(0);
  const [dimensions, setDimensions] = useState<[number, number]>([0, 0]);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    setTime(0);
    setDimensions([0, 0]);
    setFailed(false);
  }, [src]);
  useEffect(() => {
    let frame = 0;
    const update = () => {
      if (video.current)
        setTime(video.current.currentTime);
      frame = requestAnimationFrame(update);
    };
    frame = requestAnimationFrame(update);
    return () => cancelAnimationFrame(frame);
  }, [src]);
  const boxes = visibleObservations(tracks, time, fps);
  return <div>
    <div className="video-stage">
      <video ref={video} aria-label={label} controls src={src} preload="metadata" onLoadedMetadata={e => {
        setDimensions([e.currentTarget.videoWidth, e.currentTarget.videoHeight]);
        setFailed(false);
      }} onTimeUpdate={e => setTime(e.currentTarget.currentTime)} onSeeked={e => setTime(e.currentTarget.currentTime)} onError={() => setFailed(true)} />
      {dimensions[0] > 0 && <svg className="track-overlay" aria-label={`${label} track overlays`} viewBox={`0 0 ${dimensions[0]} ${dimensions[1]}`} preserveAspectRatio="xMidYMid meet">
        {boxes.map(({ trackId, observation: o }) => {
          const [x1, y1, x2, y2] = o.bbox;
          return <g key={trackId}>
            <rect x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill="none" stroke="#65dec1" strokeWidth={Math.max(2, dimensions[0] / 500)} />
            <text x={x1 + 3} y={Math.max(18, y1 - 6)} fill="#fff" stroke="#101721" strokeWidth={2} paintOrder="stroke" fontSize={Math.max(16, dimensions[0] / 65)}>
              {trackId} · {o.confidence.toFixed(2)}
            </text>
          </g>;
        })}
      </svg>}
    </div>
    {failed && <p role="alert" className="error">Browser cannot play this video. Use a browser-compatible H.264 MP4; decoding for inference is independent.</p>}
    {tracks.length > 0 && <p>Overlay uses observed track boxes at video-relative time {time.toFixed(2)}s. IDs are camera-local; blank frames may contain untracked or missed motorcycles.</p>}
  </div>;
}
