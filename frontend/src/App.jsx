import { useState, useEffect, useCallback } from "react";
import { ShieldCheck, Play, RotateCcw, Loader2 } from "lucide-react";
import "./App.css";
import Waveform from "./components/Waveform";
import ResultCard from "./components/ResultCard";
import Scoreboard from "./components/Scoreboard";
import ModelStatsPanel from "./components/ModelStatsPanel";
import { testRandomDriver, getScoreboard, resetScoreboard, getModelStats } from "./api/carvis";

export default function App() {
  const [result, setResult] = useState(null);
  const [scoreboard, setScoreboardData] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [waveState, setWaveState] = useState("idle");

  const loadScoreboard = useCallback(async () => {
    try {
      const data = await getScoreboard();
      setScoreboardData(data);
    } catch (e) {
      // silent -- scoreboard is non-critical
    }
  }, []);

  const loadStats = useCallback(async () => {
    try {
      const data = await getModelStats();
      setStats(data);
    } catch (e) {
      // backend might not have stats yet, that's ok
    }
  }, []);

  useEffect(() => {
    loadScoreboard();
    loadStats();
  }, [loadScoreboard, loadStats]);

  async function handleTest() {
    setLoading(true);
    setError(null);
    setWaveState("idle");

    try {
      const data = await testRandomDriver();
      setResult(data);
      setWaveState(data.prediction.label);
      await loadScoreboard();
    } catch (e) {
      setError(
        "Could not reach the CARVIS backend. Make sure FastAPI is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    await resetScoreboard();
    await loadScoreboard();
    setResult(null);
    setWaveState("idle");
  }

  const intensity = result
    ? Math.min(1, (result.sensor_window.accel_x_std + result.sensor_window.yaw_std) / 3)
    : 0.3;

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <ShieldCheck size={22} className="brand-icon" />
          <div>
            <div className="brand-title">CARVIS</div>
            <div className="brand-subtitle">Continuous Automotive Real-time Vehicle Intrusion System</div>
          </div>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          live demo
        </div>
      </header>

      <main className="app-main">
        <section className="hero-panel">
          <div className="hero-text">
            <div className="eyebrow">Behavioral fingerprint detection</div>
            <h1>
              Your car doesn't need to know <em>where</em> it is.<br />
              It needs to know <em>who's driving it.</em>
            </h1>
            <p>
              CARVIS learns the owner's unique driving signature -- acceleration rhythm,
              braking sharpness, turning style -- from real recorded driving data, and flags
              the moment someone else takes the wheel. No GPS. No hardware key. Just behavior.
            </p>

            <div className="hero-actions">
              <button className="btn-primary" onClick={handleTest} disabled={loading}>
                {loading ? <Loader2 size={18} className="spin" /> : <Play size={18} />}
                {loading ? "Analyzing window..." : "Test Random Driver"}
              </button>
              <button className="btn-secondary" onClick={handleReset}>
                <RotateCcw size={16} />
                Reset session
              </button>
            </div>

            {error && <div className="error-banner">{error}</div>}
          </div>

          <div className="hero-wave">
            <Waveform state={waveState} intensity={intensity} height={160} />
            <div className="wave-caption">
              {waveState === "idle" && "awaiting input -- click Test Random Driver"}
              {waveState === "owner" && "signature matched -- driving pattern consistent"}
              {waveState === "intruder" && "anomaly detected -- pattern deviates from owner"}
            </div>
          </div>
        </section>

        {scoreboard && <Scoreboard data={scoreboard} />}

        {result && (
          <section className="result-section">
            <ResultCard result={result} />
          </section>
        )}

        {stats && (
          <section className="stats-section">
            <h2 className="section-title">Model Performance</h2>
            <ModelStatsPanel stats={stats} />
          </section>
        )}

        <section className="how-it-works">
          <h2 className="section-title">How this works</h2>
          <div className="steps-grid">
            <div className="step">
              <span className="step-tag">input</span>
              <p>Real accelerometer &amp; GPS data recorded from actual drivers (UAH-DriveSet)</p>
            </div>
            <div className="step">
              <span className="step-tag">features</span>
              <p>Raw sensor noise converted into behavior signals -- braking sharpness, turn rhythm, speed consistency</p>
            </div>
            <div className="step">
              <span className="step-tag">model</span>
              <p>XGBoost classifier trained on the owner's driving fingerprint vs all other drivers in UAH-DriveSet</p>
            </div>
            <div className="step">
              <span className="step-tag">output</span>
              <p>Live verdict: does this 5-second window match the owner, or not?</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="app-footer">
        Built with XGBoost &middot; FastAPI &middot; React — trained on real UAH-DriveSet telemetry
      </footer>
    </div>
  );
}
