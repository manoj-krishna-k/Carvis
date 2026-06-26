import { Shuffle, Crown, User, Loader2 } from "lucide-react";

/**
 * DriverSelector — lets the user feed a specific driver into the model
 * instead of relying on a blind random draw. The owner (D1) is marked
 * with a crown; everyone else is a potential intruder.
 */
export default function DriverSelector({
  drivers,
  ownerDriverId,
  selected,
  onSelect,
  onRandom,
  loading,
}) {
  return (
    <section className="selector-panel">
      <div className="selector-head">
        <div>
          <h2 className="section-title selector-title">Choose a driver to analyze</h2>
          <p className="selector-sub">
            Pick any recorded driver as input, or let the system draw a blind random one.
            The model only ever learned the owner&apos;s fingerprint.
          </p>
        </div>
        <button className="btn-primary" onClick={onRandom} disabled={loading}>
          {loading ? <Loader2 size={16} className="spin" /> : <Shuffle size={16} />}
          {loading ? "Analyzing…" : "Test Random Driver"}
        </button>
      </div>

      <div className="driver-chips">
        {drivers.map((d) => {
          const isOwner = d.driver_id === ownerDriverId;
          const isActive = d.driver_id === selected;
          return (
            <button
              key={d.driver_id}
              className={`driver-chip ${isOwner ? "chip-owner" : "chip-other"} ${
                isActive ? "chip-active" : ""
              }`}
              onClick={() => onSelect(d.driver_id)}
              disabled={loading}
            >
              <span className="chip-icon">
                {isOwner ? <Crown size={15} /> : <User size={15} />}
              </span>
              <span className="chip-id">{d.driver_id}</span>
              <span className="chip-role">{isOwner ? "owner" : "other driver"}</span>
              <span
                className={`chip-dev ${
                  d.overall_deviation <= 1 ? "dev-low" : d.overall_deviation <= 2.6 ? "dev-mid" : "dev-high"
                }`}
              >
                {d.overall_deviation.toFixed(1)}σ
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
