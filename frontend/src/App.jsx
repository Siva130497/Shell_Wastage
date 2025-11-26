import { useState } from "react";

function App() {
  const [detailsFile, setDetailsFile] = useState(null);
  const [adjustmentFile, setAdjustmentFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [downloadUrl, setDownloadUrl] = useState(null);

  const backendUrl = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setDownloadUrl(null);

    if (!detailsFile || !adjustmentFile) {
      setError("Please upload both PDFs.");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("details_pdf", detailsFile);
      formData.append("adjustment_pdf", adjustmentFile);

      const res = await fetch(`${backendUrl}/generate-report`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || "Failed to generate report");
      }

      // Expecting PDF binary
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      setDownloadUrl(url);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (setter) => (e) => {
    const file = e.target.files?.[0];
    setter(file || null);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        background: "#0f172a",
        color: "#e5e7eb",
        fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        padding: "1.5rem",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "720px",
          background: "#020617",
          borderRadius: "1rem",
          padding: "2rem",
          boxShadow: "0 20px 40px rgba(0,0,0,0.6)",
          border: "1px solid #1f2937",
        }}
      >
        <h1
          style={{
            fontSize: "1.8rem",
            fontWeight: 700,
            marginBottom: "0.5rem",
            textAlign: "center",
          }}
        >
          Wastage Report Generator
        </h1>
        <p
          style={{
            fontSize: "0.95rem",
            color: "#9ca3af",
            textAlign: "center",
            marginBottom: "1.5rem",
          }}
        >
          Upload your <strong>Stock Wastage PDF</strong> and{" "}
          <strong>Adjustment / Cashier Override PDF</strong>. I’ll generate a
          merged, colour-coded analysis PDF for you.
        </p>

        <form onSubmit={handleSubmit}>
          <div style={{ marginBottom: "1rem" }}>
            <label style={{ display: "block", marginBottom: "0.35rem", fontWeight: 600 }}>
              Stock Wastage PDF
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange(setDetailsFile)}
              style={{
                width: "100%",
                padding: "0.4rem",
                background: "#020617",
                color: "#e5e7eb",
                borderRadius: "0.5rem",
                border: "1px solid #374151",
              }}
            />
            {detailsFile && (
              <p style={{ fontSize: "0.8rem", marginTop: "0.25rem", color: "#9ca3af" }}>
                Selected: {detailsFile.name}
              </p>
            )}
          </div>

          <div style={{ marginBottom: "1.5rem" }}>
            <label style={{ display: "block", marginBottom: "0.35rem", fontWeight: 600 }}>
              Adjustment / Cashier Override PDF
            </label>
            <input
              type="file"
              accept="application/pdf"
              onChange={handleFileChange(setAdjustmentFile)}
              style={{
                width: "100%",
                padding: "0.4rem",
                background: "#020617",
                color: "#e5e7eb",
                borderRadius: "0.5rem",
                border: "1px solid #374151",
              }}
            />
            {adjustmentFile && (
              <p style={{ fontSize: "0.8rem", marginTop: "0.25rem", color: "#9ca3af" }}>
                Selected: {adjustmentFile.name}
              </p>
            )}
          </div>

          {error && (
            <div
              style={{
                marginBottom: "1rem",
                padding: "0.6rem 0.8rem",
                borderRadius: "0.5rem",
                background: "rgba(248, 113, 113, 0.15)",
                border: "1px solid rgba(248, 113, 113, 0.4)",
                color: "#fecaca",
                fontSize: "0.85rem",
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            style={{
              width: "100%",
              padding: "0.7rem 1rem",
              borderRadius: "999px",
              border: "none",
              fontWeight: 600,
              fontSize: "0.95rem",
              cursor: loading ? "wait" : "pointer",
              background: loading
                ? "linear-gradient(to right, #4b5563, #6b7280)"
                : "linear-gradient(to right, #22c55e, #22d3ee)",
              color: "#020617",
              transition: "transform 0.1s ease, box-shadow 0.1s ease",
              boxShadow: loading
                ? "0 0 0 rgba(0,0,0,0)"
                : "0 15px 30px rgba(34, 197, 94, 0.35)",
            }}
          >
            {loading ? "Generating report…" : "Generate Report"}
          </button>
        </form>

        {downloadUrl && !loading && (
          <div
            style={{
              marginTop: "1.5rem",
              padding: "0.9rem 1rem",
              borderRadius: "0.75rem",
              background: "rgba(34, 197, 94, 0.1)",
              border: "1px solid rgba(34, 197, 94, 0.6)",
              textAlign: "center"
            }}
          >
            <p
              style={{
                margin: 0,
                marginBottom: "0.4rem",
                fontWeight: 600,
                color: "#bbf7d0",
              }}
            >
              Report ready ✅
            </p>
            <a
              href={downloadUrl}
              download="Wastage_analysis.pdf"
              style={{
                display: "inline-block",
                marginTop: "0.25rem",
                padding: "0.45rem 0.9rem",
                borderRadius: "999px",
                background: "#22c55e",
                color: "#022c22",
                fontWeight: 600,
                fontSize: "0.85rem",
                textDecoration: "none",
              }}
            >
              Download PDF
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;