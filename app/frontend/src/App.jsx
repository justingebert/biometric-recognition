import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const streamRef = useRef(null);
  const [persons, setPersons] = useState([]);
  const [person, setPerson] = useState("");
  const [testImages, setTestImages] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null); // image shown instead of the live cam
  const [detThr, setDetThr] = useState(0.5); // per-image match cutoff
  const [verThr, setVerThr] = useState(0.5); // fraction of gallery matches to accept
  const [camReady, setCamReady] = useState(false); // stream has frames to capture
  const [camError, setCamError] = useState(false); // webcam could not be acquired

  // Attach the acquired stream to the current <video> element (if mounted).
  function attachStream() {
    const v = videoRef.current;
    if (v && streamRef.current && v.srcObject !== streamRef.current) {
      v.srcObject = streamRef.current;
    }
  }

  // Acquire the webcam once on mount; stop tracks on unmount.
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        streamRef.current = stream;
        attachStream();
      })
      .catch(() => setCamError(true));
    return () => streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  // Re-attach the stream whenever we return to the live camera view.
  useEffect(() => {
    if (!preview) {
      setCamReady(false);
      attachStream();
    }
  }, [preview]);

  // Load enrolled persons and staged test probes.
  useEffect(() => {
    fetch(`${API}/persons`)
      .then((r) => r.json())
      .then((d) => {
        setPersons(d.persons);
        if (d.persons.length) setPerson(d.persons[0]);
      })
      .catch(() => setError("Could not reach backend. Is it running?"));
    fetch(`${API}/test-images`)
      .then((r) => r.json())
      .then((d) => setTestImages(d.images))
      .catch(() => {});
  }, []);

  function reset() {
    setResult(null);
    setError(null);
  }

  // Send a JPEG/image blob to the backend for the selected person.
  async function sendToVerify(blob) {
    setLoading(true);
    reset();
    try {
      const form = new FormData();
      form.append("person", person);
      form.append("file", blob, "input.jpg");
      form.append("detection_threshold", detThr);
      form.append("verification_threshold", verThr);
      const t0 = performance.now();
      const res = await fetch(`${API}/verify`, { method: "POST", body: form });
      const roundtrip_ms = performance.now() - t0;
      if (!res.ok) throw new Error();
      setResult({ ...(await res.json()), roundtrip_ms });
    } catch {
      setError("Verification request failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  function verifyCamera() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) {
      setError("Camera is not ready yet. Give it a moment.");
      return;
    }
    // Grab the current frame into a hidden canvas, then to a JPEG blob.
    setPreview(null);
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => sendToVerify(blob), "image/jpeg");
  }

  function verifyUpload(e) {
    const file = e.target.files?.[0];
    e.target.value = ""; // allow re-selecting the same file
    if (!file) return;
    setPreview(URL.createObjectURL(file));
    sendToVerify(file);
  }

  // Click a staged probe -> verify it server-side against the selected person.
  async function verifyTest(name) {
    setPreview(`${API}/test-images-static/${name}`);
    setLoading(true);
    reset();
    try {
      const t0 = performance.now();
      const res = await fetch(`${API}/verify-test`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          person,
          name,
          detection_threshold: detThr,
          verification_threshold: verThr,
        }),
      });
      const roundtrip_ms = performance.now() - t0;
      if (!res.ok) throw new Error();
      setResult({ ...(await res.json()), roundtrip_ms });
    } catch {
      setError("Verification request failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  const disabled = loading || !person;

  return (
    <div className="app">
      <h1>Face Verify</h1>

      <label className="who">
        Claimed identity:{" "}
        <select
          value={person}
          onChange={(e) => setPerson(e.target.value)}
          disabled={loading || !persons.length}
        >
          {persons.length ? (
            persons.map((p) => <option key={p}>{p}</option>)
          ) : (
            <option value="">— none enrolled —</option>
          )}
        </select>
      </label>

      {persons.length === 0 && (
        <p className="error">No enrolled persons. See the README (data setup).</p>
      )}

      <div className="thresholds">
        <label>
          <span>Match threshold (per image): {detThr.toFixed(2)}</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={detThr}
            onChange={(e) => setDetThr(parseFloat(e.target.value))}
            disabled={loading}
          />
        </label>
        <label>
          <span>Decision threshold (fraction of gallery): {verThr.toFixed(2)}</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={verThr}
            onChange={(e) => setVerThr(parseFloat(e.target.value))}
            disabled={loading}
          />
        </label>
      </div>

      {preview ? (
        <img src={preview} alt="probe" className="cam" />
      ) : (
        <div className="cam-wrap">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="cam"
            onLoadedMetadata={() => setCamReady(true)}
          />
          {camReady && !camError && (
            <div className="cam-guide">
              <span>Align your face in the box</span>
            </div>
          )}
          {!camReady && !camError && <div className="cam-overlay">Starting camera…</div>}
          {camError && <div className="cam-overlay">Camera unavailable</div>}
        </div>
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="actions">
        <button onClick={verifyCamera} disabled={disabled || !camReady || !!preview}>
          {loading
            ? "Verifying…"
            : camError
            ? "Camera unavailable"
            : camReady
            ? "Capture & Verify"
            : "Starting camera…"}
        </button>
        <button onClick={() => fileRef.current?.click()} disabled={disabled}>
          Upload image
        </button>
        {preview && (
          <button
            className="ghost"
            onClick={() => {
              setPreview(null);
              reset();
            }}
            disabled={loading}
          >
            Back to camera
          </button>
        )}
        <input
          ref={fileRef}
          type="file"
          accept="image/*"
          onChange={verifyUpload}
          style={{ display: "none" }}
        />
      </div>

      {testImages.length > 0 && (
        <div className="probes">
          <span className="probes-label">Test probes (click to verify):</span>
          <div className="probe-grid">
            {testImages.map((name) => (
              <button
                key={name}
                className="probe"
                onClick={() => verifyTest(name)}
                disabled={disabled}
                title={name}
              >
                <img src={`${API}/test-images-static/${name}`} alt={name} />
              </button>
            ))}
          </div>
        </div>
      )}

      {error && <p className="error">{error}</p>}

      {result && (
        <div className={`result ${result.verified ? "ok" : "no"}`}>
          {result.verified ? "✅ Verified" : "❌ Not verified"}
          <small>
            as {result.person} · {result.detections}/{result.total} matches ·{" "}
            {Math.round(result.confidence * 100)}%
          </small>
          <small className="timing">
            inference {result.inference_ms} ms ({result.inference_ms_per_comparison} ms/img
            × {result.total})
            {result.roundtrip_ms != null &&
              ` · round-trip ${Math.round(result.roundtrip_ms)} ms`}
          </small>
        </div>
      )}
    </div>
  );
}
