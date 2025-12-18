import { useEffect, useMemo, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:8000";

const defaultSample = {
  gender: "Female",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "Yes",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "Yes",
  StreamingMovies: "Yes",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 79.85,
  TotalCharges: 985.5
};

export default function Home() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [metrics, setMetrics] = useState(null);
  const [featuresText, setFeaturesText] = useState(JSON.stringify(defaultSample, null, 2));
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/models`);
        const j = await r.json();
        setModels(j.models || []);
        setSelectedModel(j.best_model || (j.models?.[0] ?? ""));
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const r = await fetch(`${API_BASE}/metrics`);
        const j = await r.json();
        setMetrics(j);
      } catch (e) {
        setError(String(e));
      }
    })();
  }, []);

  const selectedMetrics = useMemo(() => {
    if (!metrics?.models || !selectedModel) return null;
    return metrics.models[selectedModel] || null;
  }, [metrics, selectedModel]);

  const recommendedModelKey = "logistic_regression";
  const recommendedMetrics = useMemo(() => {
    if (!metrics?.models) return null;
    return metrics.models[recommendedModelKey] || null;
  }, [metrics]);

  async function runPredict() {
    setError("");
    setPrediction(null);
    try {
      const features = JSON.parse(featuresText);
      const r = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: selectedModel, features })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j?.detail || "Prediction failed");
      setPrediction(j);
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div style={{ fontFamily: "Arial, sans-serif", padding: 24, maxWidth: 1100, margin: "0 auto" }}>
      <h1>Telco Churn — Multi-Model Inference</h1>
      <p style={{ color: "#555" }}>
        Explore customer churn predictions and compare how different models identify at-risk customers.
      </p>

      {error ? (
        <div style={{ background: "#ffe7e7", border: "1px solid #ffbcbc", padding: 12, marginBottom: 12 }}>
          <b>Error:</b> {error}
        </div>
      ) : null}

      <div style={{ border: "1px solid #cde4ff", background: "#f6f9ff", borderRadius: 12, padding: 16, marginBottom: 18 }}>
        <h2 style={{ marginTop: 0, marginBottom: 6 }}>Recommended Model</h2>
        <p style={{ margin: 0, color: "#42526e" }}>
          Logistic regression is recommended as the primary model based on validation recall, while other models are shown for comparison.
        </p>
        {recommendedMetrics ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 12 }}>
            <MetricCard title="Validation (Logistic Regression)" data={recommendedMetrics.validation} />
            <MetricCard title="Test (Logistic Regression)" data={recommendedMetrics.test} />
          </div>
        ) : (
          <p style={{ color: "#777", marginTop: 10 }}>Metrics not available yet. Train models to see details.</p>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>1) Choose Model</h2>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            style={{ padding: 8, width: "100%", maxWidth: 420 }}
          >
            {models.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>

          <h3>Metrics (Validation + Test)</h3>
          {!selectedMetrics ? (
            <p style={{ color: "#777" }}>Metrics not available yet. Train models first.</p>
          ) : (
            <div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <MetricCard title="Validation" data={selectedMetrics.validation} />
                <MetricCard title="Test" data={selectedMetrics.test} />
              </div>
              <p style={{ color: "#777", marginTop: 10 }}>
                Best params (GridSearch): <code>{JSON.stringify(selectedMetrics.best_params)}</code>
              </p>
            </div>
          )}

          <h3>Model Comparison Plot</h3>
          <img
            src={`${API_BASE}/plots/model_comparison`}
            alt="Model Comparison"
            style={{ width: "100%", border: "1px solid #eee", borderRadius: 8 }}
            onError={() => {}}
          />

          <p style={{ marginTop: 10 }}>
            <a href={`${API_BASE}/eda/report`} target="_blank" rel="noreferrer">Open EDA report</a>
          </p>
        </div>

        <div style={{ border: "1px solid #ddd", borderRadius: 10, padding: 14 }}>
          <h2 style={{ marginTop: 0 }}>2) Predict</h2>
          <p style={{ color: "#777" }}>
            Paste a single customer features JSON (no <code>Churn</code>).
          </p>

          <textarea
            value={featuresText}
            onChange={(e) => setFeaturesText(e.target.value)}
            rows={22}
            style={{ width: "100%", fontFamily: "monospace", fontSize: 12, padding: 10 }}
          />

          <button onClick={runPredict} style={{ marginTop: 10, padding: "10px 14px" }}>
            Predict with {selectedModel || "model"}
          </button>

          {prediction ? (
            <div style={{ marginTop: 14, background: "#f5fff5", border: "1px solid #bfe5bf", padding: 12 }}>
              <b>Result</b>
              <pre style={{ margin: 0 }}>{JSON.stringify(prediction, null, 2)}</pre>
            </div>
          ) : null}
        </div>
      </div>

    </div>
  );
}

function MetricCard({ title, data }) {
  if (!data) return null;
  return (
    <div style={{ border: "1px solid #eee", borderRadius: 10, padding: 12 }}>
      <h4 style={{ marginTop: 0 }}>{title}</h4>
      <ul style={{ margin: 0, paddingLeft: 18 }}>
        <li>Accuracy: <b>{data.accuracy.toFixed(4)}</b></li>
        <li>Precision: <b>{data.precision.toFixed(4)}</b></li>
        <li>Recall: <b>{data.recall.toFixed(4)}</b></li>
      </ul>
    </div>
  );
}
