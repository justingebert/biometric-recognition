import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://localhost:8000";

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileRef = useRef(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null); // object URL of uploaded image

  // Start the webcam once on mount.
  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      })
      .catch(() => setError("Could not access webcam."));
  }, []);

  // Send a JPEG/image blob to the backend and show the result.
  async function sendToVerify(blob) {
    setLoading(true);
    setResult(null);
    setError(null);
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
  }

  function verifyCamera() {
    // Grab the current frame into a hidden canvas, then to a JPEG blob.
    setPreview(null);
    const video = videoRef.current;
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

  return (
    <div className="app">
      <h1>Face Verify</h1>

      {preview ? (
        <img src={preview} alt="uploaded" className="cam" />
      ) : (
        <video ref={videoRef} autoPlay playsInline muted className="cam" />
      )}
      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="actions">
        <button onClick={verifyCamera} disabled={loading}>
          {loading ? "Verifying…" : "Capture & Verify"}
        </button>
        <button onClick={() => fileRef.current?.click()} disabled={loading}>
          Upload image
        </button>
        {preview && (
          <button
            className="ghost"
            onClick={() => {
              setPreview(null);
              setResult(null);
              setError(null);
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
