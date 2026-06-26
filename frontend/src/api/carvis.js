const API_BASE = "http://127.0.0.1:8000";

export async function testRandomDriver() {
  const res = await fetch(`${API_BASE}/api/test-random`);
  if (!res.ok) throw new Error("Failed to fetch test result");
  return res.json();
}

export async function getScoreboard() {
  const res = await fetch(`${API_BASE}/api/scoreboard`);
  if (!res.ok) throw new Error("Failed to fetch scoreboard");
  return res.json();
}

export async function resetScoreboard() {
  const res = await fetch(`${API_BASE}/api/scoreboard/reset`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to reset scoreboard");
  return res.json();
}

export async function getModelStats() {
  const res = await fetch(`${API_BASE}/api/model-stats`);
  if (!res.ok) throw new Error("Failed to fetch model stats");
  return res.json();
}

export async function getDrivers() {
  const res = await fetch(`${API_BASE}/api/drivers`);
  if (!res.ok) throw new Error("Failed to fetch drivers");
  return res.json();
}

export async function testDriver(driverId) {
  const res = await fetch(`${API_BASE}/api/test-driver?driver_id=${encodeURIComponent(driverId)}`);
  if (!res.ok) throw new Error("Failed to fetch driver result");
  return res.json();
}
