import test from "node:test";
import assert from "node:assert/strict";
import { DEFAULT_TIMEOUT_MS, normalizeTimeoutMs, safeFailure, normalizeBaseUrl } from "../helpers.js";
import { registerFusionMemoryTools } from "../runtime.js";

test("safeFailure hides raw errors", () => {
  const result = safeFailure(new Error("connect ECONNREFUSED 127.0.0.1:8765"));
  assert.equal(result.content[0].type, "text");
  assert.match(result.content[0].text, /fusion-memory doctor/);
  assert.doesNotMatch(result.content[0].text, /ECONNREFUSED/);
});

test("normalizeBaseUrl trims trailing slash", () => {
  assert.equal(normalizeBaseUrl("http://127.0.0.1:8765/"), "http://127.0.0.1:8765");
});

test("normalizeTimeoutMs falls back for invalid values", () => {
  assert.equal(normalizeTimeoutMs(undefined), DEFAULT_TIMEOUT_MS);
  assert.equal(normalizeTimeoutMs("not-a-number"), DEFAULT_TIMEOUT_MS);
  assert.equal(normalizeTimeoutMs(0), DEFAULT_TIMEOUT_MS);
  assert.equal(normalizeTimeoutMs(-50), DEFAULT_TIMEOUT_MS);
});

test("normalizeTimeoutMs clamps runtime bounds", () => {
  assert.equal(normalizeTimeoutMs(20), 100);
  assert.equal(normalizeTimeoutMs("2500"), 2000);
  assert.equal(normalizeTimeoutMs(750), 750);
});

test("registerFusionMemoryTools exposes store and search handlers", async () => {
  const calls = [];
  const previousFetch = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    calls.push({url: String(url), body: JSON.parse(String(options.body))});
    return {
      ok: true,
      async json() {
        return String(url).endsWith("/answer-context")
          ? {source_spans: [{text: "runtime smoke token"}]}
          : {span_id: "span-1"};
      },
    };
  };
  try {
    const tools = new Map();
    registerFusionMemoryTools({
      registerTool(factory, options) {
        const tool = factory({
          workspaceId: "ws",
          userId: "u",
          sessionId: "s",
          pluginConfig: {baseUrl: "http://memory.local", timeoutMs: 500},
        });
        for (const name of options.names) {
          tools.set(name, tool);
        }
      },
    });

    await tools.get("fusion_memory_store").execute("write", {content: "remember this"});
    await tools.get("fusion_memory_search").execute("search", {query: "remember", limit: 3});

    assert.ok(tools.has("fusion_memory_store"));
    assert.ok(tools.has("fusion_memory_search"));
    assert.equal(new URL(calls[0].url).pathname, "/add");
    assert.equal(new URL(calls[1].url).pathname, "/answer-context");
    assert.equal(calls[0].body.metadata.source, "openclaw-tool");
    assert.equal(calls[1].body.budget.allow_cross_session, true);
  } finally {
    globalThis.fetch = previousFetch;
  }
});
