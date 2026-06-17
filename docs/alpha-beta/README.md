# Alpha/Beta Simulation

Run local alpha checks:

```bash
fusion-memory alpha-test --report docs/alpha-beta/alpha-latest.json
```

Run beta readiness checks:

```bash
fusion-memory beta-test --report docs/alpha-beta/beta-latest.json
```

The reports never include model API keys. Test model configuration may be passed
by path with `/public/home/wwb/test_key/key.txt` when benchmark commands need it.
