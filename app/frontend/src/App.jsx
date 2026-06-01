import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Start the webcam once on mount.
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setError("Could not access webcam."));
  }, []);

  async function verify() {
    setLoading(true);
    setResult(null);
    setError(null);

    // Grab the current frame into a hidden canvas, then to a JPEG blob.
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    canvas.toBlob(async (blob) => {
      try {
        const form = new FormData();
        form.append("file", blob, "input.jpg");
        const res = await fetch(`${API}/verify`, { method: "POST", body: form });
        if (!res.ok) throw new Error();
        setResult(await res.json());
      } catch {
        setError("Verification request failed. Is the backend running?");
      } finally {
        setLoading(false);
      }
    }, "image/jpeg");
  }

  return (
    <div className="app">
      <h1>Face Verify</h1>
      <video ref={videoRef} autoPlay playsInline muted className="cam" />
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <button onClick={verify} disabled={loading}>
        {loading ? "Verifying…" : "Verify"}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className={`result ${result.verified ? "ok" : "no"}`}>
          {result.verified ? "✅ Verified" : "❌ Not verified"}
          <small>
            {result.detections}/{result.total} matches ·{" "}
            {Math.round(result.confidence * 100)}%
          </small>
        </div>
      )}
    </div>
  );
}
