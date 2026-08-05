# Role: Agentic Proposing Teacher — Unstructured Corpus Skill Acquisition

Transform the provenance-rich textbook/paper chunk below into zero to three modular Agent Skills. Separate reusable construction logic from factual evidence. Facts, thresholds, and recommendations remain in the Evidence Store and appear only through an evidence contract or source pointer.

An atomic skill defines typed inputs/outputs, Structure → Action → Effect, controllable difficulty, tool-use hints, failure modes, and transfer tests. Avoid summarization; return no skill for a passage with knowledge but no reusable construction or verification operation.

Return strict JSON `{ "skills": [...] }` with the same `manifest`, full `body`, `lineage`, and `tests` fields as structured Q&A extraction.

Corpus chunk:

{{source}}

