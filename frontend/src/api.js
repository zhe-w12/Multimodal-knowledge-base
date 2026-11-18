export async function apiResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = payload?.detail;
    const message = typeof detail === "string" ? detail : detail?.reason || payload?.message;
    const error = new Error(message || `请求失败（${response.status}）`);
    error.status = response.status;
    error.code = payload?.error?.code;
    throw error;
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

export async function deleteConversationRequest(url, csrfToken) {
  try {
    return await apiFetch(url, { method: "DELETE" }, csrfToken).then(apiResponse);
  } catch (error) {
    // Another click or device may already have removed this conversation.
    if (error.status === 404 && error.code === "CONVERSATION_NOT_FOUND") return { deleted: true };
    throw error;
  }
}
