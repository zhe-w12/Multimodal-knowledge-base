import assert from "node:assert/strict";
import test from "node:test";
import { apiResponse, deleteConversationRequest } from "./api.js";

test("API errors retain status and resource code", async () => {
  await assert.rejects(apiResponse(new Response(JSON.stringify({
    detail: "该对话已删除或不可访问。", error: { code: "CONVERSATION_NOT_FOUND" },
  }), { status: 404 })), (error) => error.status === 404 && error.code === "CONVERSATION_NOT_FOUND");
});

for (const [status, code, succeeds] of [
  [200, null, true], [404, "CONVERSATION_NOT_FOUND", true],
  [404, "API_NOT_FOUND", false], [401, "HTTP_401", false],
  [403, "HTTP_403", false], [503, "SERVER_ERROR", false],
]) {
  test(`delete handles ${status} ${code} without hiding unrelated failures`, async (t) => {
    t.mock.method(globalThis, "fetch", async (_url, init) => {
      assert.equal(init.method, "DELETE");
      assert.equal(init.credentials, "include");
      assert.equal(init.headers.get("X-CSRF-Token"), "test-csrf");
      return new Response(JSON.stringify(status === 200 ? { deleted: true } : {
        detail: "test failure", error: { code },
      }), { status });
    });
    if (succeeds) assert.deepEqual(await deleteConversationRequest("/api/conversations/test", "test-csrf"), { deleted: true });
    else await assert.rejects(deleteConversationRequest("/api/conversations/test", "test-csrf"), (error) => error.status === status && error.code === code);
  });
}
