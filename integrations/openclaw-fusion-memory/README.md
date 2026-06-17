# Fusion Memory for OpenClaw

This external OpenClaw plugin connects OpenClaw to the local Fusion Memory service.

Install with:

```bash
fusion-memory install-agent --target openclaw
```

Manual local install:

```bash
openclaw plugins install --link /public/home/wwb/memory/integrations/openclaw-fusion-memory
openclaw gateway restart
```

If the tool says Fusion Memory is not available, run:

```bash
fusion-memory doctor
```
