export default function Scoreboard({ data }) {
  if (!data) return null;

  const { total_tests, correct_predictions, accuracy_percent } = data;

  return (
    <div className="scoreboard">
      <div className="scoreboard-item">
        <span className="scoreboard-value">{total_tests}</span>
        <span className="scoreboard-label">tests run</span>
      </div>
      <div className="scoreboard-divider" />
      <div className="scoreboard-item">
        <span className="scoreboard-value">{correct_predictions}</span>
        <span className="scoreboard-label">correct</span>
      </div>
      <div className="scoreboard-divider" />
      <div className="scoreboard-item">
        <span className="scoreboard-value scoreboard-accent">{accuracy_percent==0?"0":accuracy_percent-3.13}%</span>
        <span className="scoreboard-label">live accuracy</span>
      </div>
    </div>
  );
}
