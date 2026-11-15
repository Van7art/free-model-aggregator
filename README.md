# free-model-aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Van7art/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Van7art/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A deterministic aggregator that routes OpenAI and Anthropic chat requests to the best free model across ten simulated providers, with failover on 402/429 and faithful translation across the OpenAI, Anthropic and Gemini wires.

## Run it

```sh
MAKEFUN_PROMPT_FILE=brief.txt MAKEFUN_ENV_URL=http://localhost:8080 node main.ts
```

`node main.ts --list` prints the probed provider catalog (the same document the server registers) and exits.

## How it works

- **Routing.** Every request is served by the highest-rated free model that meets its requirements (tools, vision, JSON mode, minimum context), ranked by rating less a small load penalty per call already made to that provider, so traffic spreads over the top providers instead of draining one provider's per-trial budget.
- **Failover.** A provider that refuses (402, 429 or 5xx) is retried once — a refused call is usually one numbered call in that provider's schedule — then the request fails over to the next candidate.
- **Wires.** Requests are normalised to one canonical form (`lib/upstream.ts`) and rebuilt per upstream wire, so the system prompt, user text, tool names in order, output cap, stop sequences, image bytes and JSON-mode flag arrive at the provider exactly as the client sent them. Streams are buffered and re-emitted from the parsed answer, so usage and stop reasons are exact in both stream shapes.
- **JSON mode.** A request in explicit JSON mode only goes to an OpenAI-wire or Gemini-wire provider: the Anthropic wire has no JSON-mode field, so it could not echo the flag back.
- **Determinism.** No clock, no unseeded randomness. The same trial replays byte for byte.

## Layout

- `main.ts` — the server: brief, catalog probe, register/poll loop, routing, handlers.
- `lib/providers.ts` — the brief's three documents, the catalog probe and the catalog build.
- `lib/upstream.ts` — the canonical request, the three upstream wire builders and answer parsers, and the two inbound answer shapes.
- `lib/wire.ts` — model listings and error envelopes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=4e96xqxt&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
