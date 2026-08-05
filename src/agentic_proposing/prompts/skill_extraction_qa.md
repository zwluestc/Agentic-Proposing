# Role: Agentic Proposing Teacher — Structured Q&A Skill Acquisition

Reverse-engineer the benchmark item below into zero or more **atomic reusable problem-construction skills**.

Each skill must preserve the paper's tuple: intent, executable construction method, difficulty effect, and tool hint. Do not copy surface entities, numbers, diagnoses, answers, or wording. A question may yield several reasoning nodes, but return no skill when it contains only a fact and no reusable construction operation.

Return strict JSON `{ "skills": [...] }`. Every item must contain:

- `manifest`: `skill_id`, `name`, `version`, `kind=atomic`, `status=candidate`, `description`, `intent`, triggers, typed inputs/outputs, difficulty effect, tools, evidence contract, source refs, metrics, tags.
- `body`: full `SKILL.md` with Structure, Action, Effect, failure handling, and output contract.
- `lineage`: reasoning nodes and abstractions.
- `tests`: positive, negative, and cross-domain transfer JSON rows.

Source item:

{{source}}

