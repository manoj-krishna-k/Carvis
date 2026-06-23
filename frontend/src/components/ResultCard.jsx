import { ShieldCheck, ShieldAlert, Gauge, Activity, Navigation } from "lucide-react";

export default function ResultCard({ result }) {
  if (!result) return null;

  const { sensor_window, prediction, ground_truth, was_correct } = result;
  const isOwner = prediction.label === "owner";

  return (
    <div className={`result-card ${isOwner ? "result-owner" : "result-intruder"}`}>
      <div className="result-header">
        <div className="result-icon">
          {isOwner ? <ShieldCheck size={28} /> : <ShieldAlert size={28} />}
        </div>
        <div>
          <div className="result-title">
            {isOwner ? "NORMAL — Matches Owner Fingerprint" : "ALERT — Unauthorized Driver Detected"}
          </div>
          <div className="result-subtitle">
            confidence score: <span className="mono-value">{prediction.confidence_score}</span>
          </div>
        </div>
      </div>

      <div className="result-metrics">
        <div className="metric">
          <Activity size={14} />
          <span className="metric-label">accel_x_std</span>
          <span className="metric-value">{sensor_window.accel_x_std}</span>
        </div>
        <div className="metric">
          <Activity size={14} />
          <span className="metric-label">accel_y_std</span>
          <span className="metric-value">{sensor_window.accel_y_std}</span>
        </div>
        <div className="metric">
          <Navigation size={14} />
          <span className="metric-label">yaw_std</span>
          <span className="metric-value">{sensor_window.yaw_std}</span>
        </div>
        <div className="metric">
          <Gauge size={14} />
          <span className="metric-label">speed</span>
          <span className="metric-value">
            {sensor_window.speed_mean !== null ? `${sensor_window.speed_mean} km/h` : "—"}
          </span>
        </div>
        <div className="metric">
          <span className="metric-label">harsh accel events</span>
          <span className="metric-value">{sensor_window.harsh_accel_count}</span>
        </div>
        <div className="metric">
          <span className="metric-label">harsh turn events</span>
          <span className="metric-value">{sensor_window.harsh_turn_count}</span>
        </div>
      </div>

      <div className="result-truth">
        <div className="truth-row">
          <span className="truth-label">Ground truth</span>
          <span className="truth-value">
            Driver {ground_truth.actual_driver_id} · {ground_truth.actual_behavior} behavior
          </span>
        </div>
        <div className={`truth-badge ${was_correct ? "truth-correct" : "truth-wrong"}`}>
          {was_correct ? "✓ Model predicted correctly" : "✗ Model got this one wrong"}
        </div>
      </div>
    </div>
  );
}
