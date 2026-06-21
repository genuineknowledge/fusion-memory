import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { registerFusionMemoryTools } from "./runtime.js";

export default definePluginEntry({
  id: "fusion-memory",
  name: "Fusion Memory",
  description: "Connects OpenClaw to a local Fusion Memory service.",
  kind: "memory",
  register: registerFusionMemoryTools,
});
