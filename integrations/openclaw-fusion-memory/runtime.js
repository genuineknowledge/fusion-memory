import { DEFAULT_BASE_URL, normalizeBaseUrl, normalizeTimeoutMs, safeFailure } from "./helpers.js";

async function postJson(baseUrl, path, payload, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error("fusion memory request failed");
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

function scopeFromContext(ctx) {
  return {
    workspace_id: ctx?.workspaceId || ctx?.agentId || "openclaw",
    user_id: ctx?.userId || process.env.USER || process.env.USERNAME || "user",
    agent_id: "openclaw",
    session_id: ctx?.sessionKey || ctx?.sessionId || "openclaw-session",
    app_id: "fusion-memory",
  };
}

function configFromContext(ctx) {
  const config = ctx?.pluginConfig || ctx?.config?.plugins?.entries?.["fusion-memory"]?.config || {};
  return {
    baseUrl: normalizeBaseUrl(process.env.FUSION_MEMORY_BASE_URL || config.baseUrl || DEFAULT_BASE_URL),
    timeoutMs: normalizeTimeoutMs(process.env.FUSION_MEMORY_TIMEOUT_MS || config.timeoutMs),
  };
}

function textResult(value) {
  return {content: [{type: "text", text: typeof value === "string" ? value : JSON.stringify(value)}]};
}

function makeTool(ctx, name, description, parameters, handler) {
  return {
    name,
    description,
    parameters,
    async execute(_toolCallId, params) {
      const cfg = configFromContext(ctx);
      const scope = scopeFromContext(ctx);
      try {
        return await handler(params || {}, cfg, scope);
      } catch (error) {
        return safeFailure(error);
      }
    },
  };
}

export function registerFusionMemoryTools(api) {
  api.registerTool((ctx) =>
    makeTool(
      ctx,
      "fusion_memory_search",
      "Search Fusion Memory for durable preferences, facts, and prior context.",
      {
        type: "object",
        properties: {query: {type: "string"}, limit: {type: "integer"}},
        required: ["query"],
        additionalProperties: false,
      },
      async (params, cfg, scope) => {
        const data = await postJson(
          cfg.baseUrl,
          "/answer-context",
          {query: String(params.query || ""), scope, budget: {limit: Number(params.limit || 8), allow_cross_session: true}},
          cfg.timeoutMs,
        );
        return textResult(data);
      },
    ),
    {names: ["fusion_memory_search"]},
  );

  api.registerTool((ctx) =>
    makeTool(
      ctx,
      "fusion_memory_get",
      "Retrieve exact Fusion Memory context for a query.",
      {
        type: "object",
        properties: {query: {type: "string"}, limit: {type: "integer"}},
        required: ["query"],
        additionalProperties: false,
      },
      async (params, cfg, scope) => {
        const data = await postJson(
          cfg.baseUrl,
          "/search",
          {query: String(params.query || ""), scope, options: {limit: Number(params.limit || 8), allow_cross_session: true}},
          cfg.timeoutMs,
        );
        return textResult(data);
      },
    ),
    {names: ["fusion_memory_get"]},
  );

  api.registerTool((ctx) =>
    makeTool(
      ctx,
      "fusion_memory_store",
      "Store a durable user preference, project fact, or stable decision in Fusion Memory.",
      {
        type: "object",
        properties: {content: {type: "string"}},
        required: ["content"],
        additionalProperties: false,
      },
      async (params, cfg, scope) => {
        const content = String(params.content || "").trim();
        if (!content) {
          return textResult("Memory content is empty.");
        }
        const data = await postJson(
          cfg.baseUrl,
          "/add",
          {input: {role: "user", content}, scope, metadata: {source: "openclaw-tool"}},
          cfg.timeoutMs,
        );
        return textResult({ok: true, saved: true, result: data});
      },
    ),
    {names: ["fusion_memory_store"]},
  );

  api.registerTool((ctx) =>
    makeTool(
      ctx,
      "fusion_memory_clear",
      "Clear Fusion Memory rows for the current OpenClaw scope when the user explicitly asks.",
      {type: "object", properties: {}, additionalProperties: false},
      async (_params, cfg, scope) => {
        const data = await postJson(cfg.baseUrl, "/clear", {scope, allow_cross_session: true}, cfg.timeoutMs);
        return textResult({ok: true, cleared: true, result: data});
      },
    ),
    {names: ["fusion_memory_clear"]},
  );
}
