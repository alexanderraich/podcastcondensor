# AGENT.md — Engineering Guide for Coding Agents

Practical habits to avoid common failures in multi-step software projects.

## 1. Reuse existing work
- Prefer existing pipelines, helpers, caches, and intermediate outputs over building new parallel systems.
- Before adding new logic, inspect what the current system already produces.
- If an existing artifact is already reduced, structured, or cached, consume it directly instead of recomputing it.

## 2. Keep expensive steps small
- Avoid feeding large inputs into expensive tools unless necessary.
- Use chunking, reduction, and caching to keep workloads manageable.
- Prefer small intermediate representations over raw source material when they are sufficient.
- Never assume a large prompt or large output is the best path.

## 3. Separate extraction from merging
- Extract small, local pieces of information first.
- Merge and deduplicate with deterministic code when possible.
- Do not ask a model to maintain large global state if ordinary code can do it more reliably.
- Keep model outputs compact and focused on new information.

## 4. Validate outputs strictly
- Assume generated outputs can be malformed, contradictory, or incomplete.
- Validate schema, types, and consistency before using results downstream.
- Add checks for contradictory labels, missing fields, and obviously bad classifications.
- Fail fast on invalid structured output and retry or repair only when needed.

## 5. Make workflows resumable
- Cache intermediate artifacts at each stage.
- Resume from the last successful step after interruptions.
- Avoid rerunning expensive work that already succeeded.
- Keep outputs written to clear, inspectable files.

## 6. Instrument bottlenecks
- Log stage timings.
- Log what was reused, what was recomputed, and what failed.
- Make slow steps visible.
- Add enough logging to identify where time is being spent.

## 7. Use the smallest working scope
- Start with one item, one block, one chunk, or one episode before scaling up.
- Support partial runs for debugging.
- Avoid large batch processing until the smaller path is verified.
- Keep the debugging surface area narrow.

## 8. Be careful with source completeness
- Do not trust partial or surface-level listings as complete.
- Verify ordering and completeness before building downstream logic.
- If a source may be incomplete, confirm it with a more reliable retrieval method.
- Add logging that shows what was found and what was missing.

## 9. Prefer simple architecture
- Use the simplest design that works.
- Avoid parallel pipelines and duplicate logic.
- Keep responsibilities separated: input, reduction, extraction, merge, validation, output.
- Choose clarity over cleverness.

## 10. Design for debugging
- Add small test modes.
- Make it easy to run only one part of a larger pipeline.
- Provide explicit output paths for generated artifacts.
- Make failures easy to reproduce and inspect.

## 11. Don't spin — compile a diagnostic on the first sign of trouble
- If a problem's solution isn't immediately obvious, do not attempt iterations.
- Do not tweak prompts, try variations, or write code hoping to stumble on a fix.
- Instead, immediately compile a thorough diagnostic prompt for the user's research agent:
  - What was attempted and why it failed (with actual data/errors)
  - The relevant code, prompt text, and validation logic (inline)
  - Hardware constraints (RAM, VRAM, models, latency)
  - What's needed from the agent (specific unanswered questions)
- Hand the prompt to the user and wait for their research agent's answer.
- "Try something and see" is not allowed for non-trivial problems — diagnose first.

## Checklist

Before claiming success, verify:

- [ ] reused existing outputs where possible
- [ ] kept expensive steps small
- [ ] separated extraction from merging (deterministic merge)
- [ ] validated outputs strictly
- [ ] made the workflow resumable
- [ ] instrumented timing and bottlenecks
- [ ] supported partial runs for debugging
- [ ] verified source completeness
- [ ] used the smallest working scope first
