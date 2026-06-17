import test from "node:test";
import assert from "node:assert/strict";
import { safeFailure, normalizeBaseUrl } from "../helpers.js";

test("safeFailure hides raw errors", () => {
  const result = safeFailure(new Error("connect ECONNREFUSED 127.0.0.1:8765"));
  assert.equal(result.content[0].type, "text");
  assert.match(result.content[0].text, /fusion-memory doctor/);
  assert.doesNotMatch(result.content[0].text, /ECONNREFUSED/);
});

test("normalizeBaseUrl trims trailing slash", () => {
  assert.equal(normalizeBaseUrl("http://127.0.0.1:8765/"), "http://127.0.0.1:8765");
});
