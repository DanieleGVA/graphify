## enterpriphy

This repository is **Enterpriphy** (formerly `graphify`). The legacy `graphify` slash command, CLI, and `graphify-out/` directory remain functional during the migration window described in [`MIGRATION_PLAN.md`](./MIGRATION_PLAN.md).

This project has a knowledge graph at `graphify-out/` (legacy name, kept for one release cycle).

Rules:
- Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` for god nodes and community structure.
- If `graphify-out/wiki/index.md` exists, navigate it instead of reading raw files.
- After modifying code files in this session, run `enterpriphy update .` (or the alias `graphify update .`) to keep the graph current (AST-only, no API cost).
- For the target v1 architecture (storage plane, parser routing, bi-temporal facts, hybrid query), see [`ARCHITECTURE_v2.md`](./ARCHITECTURE_v2.md).
