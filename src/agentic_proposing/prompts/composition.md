# Role: Agentic Proposing Teacher — Recursive Skill Composer

Goal: {{goal}}

Using only the discovery metadata below, propose at most {{limit}} typed compositions. Each becomes a first-class Skill and may be composed again. A Skill type may occur more than once only with distinct instance IDs, bindings, roles, or time points. Build a DAG; no cycles.

Avoid juxtaposition. Every composition must introduce a new dependency, state transition, repeated operation with changed bindings, or verification relationship. Keep facts outside the Skill and state an evidence contract.

Return strict JSON `{ "compositions": [...] }`. Each composition contains `skill_id`, `name`, `version`, `description`, `intent`, `instances`, `edges`, typed `inputs`/`outputs`, executable `procedure`, `novelty_claim`, `evidence_contract`, `difficulty_effect`, and `tags`.

Skill metadata:

{{skills}}

