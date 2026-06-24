export default function ModelStatsPanel({ stats }) {
  if (!stats) return null;

  const items = [
    { label: "Live Demo", value: stats.live_demo_accuracy ?? stats.trip_level_accuracy, desc: "Correct on held-out full trips (how the demo scores)" },
    { label: "ROC-AUC", value: stats.roc_auc, desc: "Overall separation between owner and intruder windows" },
    { label: "Window Accuracy", value: stats.accuracy, desc: "Single-window classification on test set" },
    { label: "Owner Recall", value: stats.recall, desc: "Of all real owner windows, how many were caught" },
  ];

  return (
    <div className="stats-panel">
      <div className="stats-grid">
        {items.map((item) => (
          <div className="stat-box" key={item.label}>
            <div className="stat-value">{(item.value * 100).toFixed(1)}%</div>
            <div className="stat-label">{item.label}</div>
            <div className="stat-desc">{item.desc}</div>
          </div>
        ))}
      </div>
      <div className="stats-footer">
        Trained on driver <span className="mono-value">{stats.owner_driver_id}</span> ·
        Tested across <span className="mono-value">{stats.num_drivers_tested}</span> drivers ·
        <span className="mono-value"> {stats.total_windows_tested}</span> total windows
      </div>
    </div>
  );
}
