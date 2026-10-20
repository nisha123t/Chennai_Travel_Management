const BASE = (import.meta.env && import.meta.env.VITE_API_URL) ?? "http://localhost:8000";

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error(`Can't reach the backend at ${BASE}. Start it with: uvicorn main:app --reload --port 8000`);
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

export const api = {
  stops: () => request("/stops"),
  routes: (payload) => request("/routes", { method: "POST", body: JSON.stringify(payload) }),
  metrics: () => request("/models/metrics"),
};
