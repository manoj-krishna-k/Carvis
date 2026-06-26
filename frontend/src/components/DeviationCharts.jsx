import {
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ReferenceLine,
  Tooltip,
} from "recharts";
import { Crosshair, GitCompareArrows } from "lucide-react";

const OWNER = "#3DDC84";
const INTRUDER = "#FF4757";
const AMBER = "#FFB454";

function devColor(sigma) {
  const a = Math.abs(sigma);
  if (a <= 1) return OWNER;
  if (a <= 2.6) return AMBER;
  return INTRUDER;
}

const tooltipStyle = {
  background: "#181C23",
  border: "1px solid #2E3440",
  borderRadius: 8,
  fontSize: 12,
  color: "#E8EAED",
};

/**
 * DeviationCharts — the visual answer to "how far is this driver from the
 * owner?". Three views:
 *   1. A radar of per-behavior deviation (in sigma units)
 *   2. A signed bar chart showing direction of each deviation
 *   3. A bar across every driver, ranking distance from the owner
 */
export default function DeviationCharts({ comparison, drivers, currentDriverId, ownerDriverId }) {
  if (!comparison) return null;

  const radarData = comparison.features.map((f) => ({
    feature: f.label,
    deviation: Number(Math.abs(f.deviation_sigma).toFixed(2)),
  }));

  const barData = comparison.features.map((f) => ({
    label: f.label,
    sigma: f.deviation_sigma,
  }));

  const overviewData = (drivers || []).map((d) => ({
    driver: d.driver_id,
    deviation: d.overall_deviation,
    isOwner: d.driver_id === ownerDriverId,
    isCurrent: d.driver_id === currentDriverId,
  }));

  const overall = comparison.overall_deviation;
  const level = overall <= 1 ? "matches owner" : overall <= 2.6 ? "drifting from owner" : "far from owner";
  const levelColor = overall <= 1 ? OWNER : overall <= 2.6 ? AMBER : INTRUDER;

  return (
    <section className="deviation-section">
      <div className="deviation-head">
        <div>
          <h2 className="section-title">
            <GitCompareArrows size={15} style={{ verticalAlign: "-2px", marginRight: 8 }} />
            How far this driver is from the owner
          </h2>
          <p className="deviation-sub">
            Each axis is a driving habit, measured in standard deviations (σ) away from the owner&apos;s
            learned fingerprint. The bigger the shape, the less it drives like the owner.
          </p>
        </div>
        <div className="deviation-score" style={{ borderColor: levelColor }}>
          <span className="deviation-score-value" style={{ color: levelColor }}>
            {overall.toFixed(2)}σ
          </span>
          <span className="deviation-score-label">{level}</span>
        </div>
      </div>

      <div className="charts-grid">
        <div className="chart-card">
          <div className="chart-title">
            <Crosshair size={13} /> Behavioral deviation profile
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <RadarChart data={radarData} outerRadius="72%">
              <PolarGrid stroke="#2E3440" />
              <PolarAngleAxis dataKey="feature" tick={{ fill: "#8B92A0", fontSize: 10 }} />
              <PolarRadiusAxis tick={{ fill: "#565D6B", fontSize: 9 }} stroke="#2E3440" />
              <Radar
                name="deviation (σ)"
                dataKey="deviation"
                stroke={levelColor}
                fill={levelColor}
                fillOpacity={0.35}
              />
              <Tooltip contentStyle={tooltipStyle} />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <div className="chart-title">Direction of each deviation (σ)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} layout="vertical" margin={{ left: 8, right: 16, top: 4, bottom: 4 }}>
              <XAxis type="number" tick={{ fill: "#8B92A0", fontSize: 10 }} stroke="#2E3440" />
              <YAxis
                type="category"
                dataKey="label"
                width={150}
                tick={{ fill: "#8B92A0", fontSize: 10 }}
                stroke="#2E3440"
              />
              <ReferenceLine x={0} stroke="#565D6B" />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="sigma" radius={[3, 3, 3, 3]}>
                {barData.map((d, i) => (
                  <Cell key={i} fill={devColor(d.sigma)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card chart-card-wide">
        <div className="chart-title">Distance from owner across all drivers</div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={overviewData} margin={{ left: 0, right: 8, top: 8, bottom: 4 }}>
            <XAxis dataKey="driver" tick={{ fill: "#8B92A0", fontSize: 11 }} stroke="#2E3440" />
            <YAxis
              tick={{ fill: "#8B92A0", fontSize: 10 }}
              stroke="#2E3440"
              label={{ value: "σ from owner", angle: -90, position: "insideLeft", fill: "#565D6B", fontSize: 10 }}
            />
            <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            <Bar dataKey="deviation" radius={[4, 4, 0, 0]}>
              {overviewData.map((d, i) => (
                <Cell
                  key={i}
                  fill={d.isOwner ? OWNER : devColor(d.deviation)}
                  stroke={d.isCurrent ? "#E8EAED" : "transparent"}
                  strokeWidth={d.isCurrent ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="chart-legend">
          <span><i style={{ background: OWNER }} /> owner baseline</span>
          <span><i style={{ background: AMBER }} /> mild drift</span>
          <span><i style={{ background: INTRUDER }} /> clear intruder</span>
          <span><i className="legend-current" /> current selection</span>
        </div>
      </div>
    </section>
  );
}
