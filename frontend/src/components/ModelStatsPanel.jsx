export default function ModelStatsPanel({ stats }) {
  if (!stats) return null;

  const items = [
    { label: "Precision", value: stats.precision, desc: "Of all 'owner' predictions, how many were right" },
    { label: "Recall", value: stats.recall, desc: "Of all real owner windows, how many were caught" },
    { label: "Accuracy", value: stats.accuracy, desc: "Overall correct predictions" },
    { label: "F1 Score", value: stats.f1_score, desc: "Balance of precision and recall" },
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
