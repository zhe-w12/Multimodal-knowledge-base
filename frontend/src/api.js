export async function apiResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.reason || payload?.message;
    throw new Error(message || `请求失败（${response.status}）`);
  }
  return payload;
}

export function apiFetch(url, init = {}, csrfToken = "") {
  const method = String(init.method || "GET").toUpperCase();
  const headers = new Headers(init.headers || {});
  if (csrfToken && ["POST", "PUT", "PATCH", "DELETE"].includes(method)) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  return fetch(url, { ...init, headers, credentials: "include" });
}
