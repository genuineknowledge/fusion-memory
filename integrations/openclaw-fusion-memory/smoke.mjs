import { registerFusionMemoryTools } from "./runtime.js";

const baseUrl = String(process.env.FUSION_MEMORY_SMOKE_MEMORY_URL || process.env.FUSION_MEMORY_BASE_URL || "http://127.0.0.1:8765");
const token = `openclaw-smoke-${Date.now()}-${Math.random().toString(16).slice(2)}`;

function smokeContext() {
  return {
    workspaceId: "agent-runtime-smoke",
    userId: process.env.USER || process.env.USERNAME || "smoke-user",
    agentId: "openclaw",
    sessionId: "agent-runtime-smoke",
    pluginConfig: {
      baseUrl,
      timeoutMs: Number(process.env.FUSION_MEMORY_TIMEOUT_MS || 1500),
    },
  };
}

function collectTools() {
  const tools = new Map();
  registerFusionMemoryTools({
    registerTool(factory, options) {
      const tool = factory(smokeContext());
      const names = options?.names || [tool.name];
      for (const name of names) {
        tools.set(name, tool);
      }
    },
  });
  return tools;
}

function textFromToolResult(result) {
  return String(result?.content?.[0]?.text || "");
}

async function main() {
  const tools = collectTools();
  const store = tools.get("fusion_memory_store");
  const search = tools.get("fusion_memory_search");
  if (!store || !search) {
    throw new Error("fusion memory plugin tools were not registered");
  }

  const writeResult = await store.execute("openclaw-smoke-write", {
    content: `OpenClaw runtime smoke token: ${token}`,
  });
  const searchResult = await search.execute("openclaw-smoke-search", {
    query: `Find OpenClaw runtime smoke token ${token}`,
    limit: 3,
  });
  const writeText = textFromToolResult(writeResult);
  const searchText = textFromToolResult(searchResult);
  const writeSmoke = writeText.includes("\"saved\":true") || writeText.includes("\"saved\": true");
  const retrieveSmoke = searchText.includes(token);
  if (!writeSmoke || !retrieveSmoke) {
    process.exitCode = 1;
  }
  console.log(
    JSON.stringify({
      write_smoke: writeSmoke,
      retrieve_smoke: retrieveSmoke,
      ok: writeSmoke && retrieveSmoke,
      message:
        writeSmoke && retrieveSmoke
          ? "OpenClaw adapter runtime smoke completed."
          : "OpenClaw adapter runtime smoke did not verify write and retrieve.",
    }),
  );
}

main().catch(() => {
  console.log(
    JSON.stringify({
      write_smoke: false,
      retrieve_smoke: false,
      ok: false,
      message: "OpenClaw adapter runtime smoke could not reach Fusion Memory through the plugin tools. Run fusion-memory doctor.",
    }),
  );
  process.exitCode = 1;
});
