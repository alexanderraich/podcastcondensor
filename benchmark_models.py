#!/usr/bin/env python3
"""Progressive gated model benchmark — live output, partial persistence, fail-fast."""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from podcastcondensor.ollama_client import generate

# ── live output ───────────────────────────────────────────────────────────────
def log(msg: str, end: str = "\n"):
    print(msg, end=end, flush=True)

def measure_vram():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        parts = r.stdout.strip().split(",")
        return int(parts[0].strip()), int(parts[1].strip())
    except Exception:
        return 0, 0

def extract_json(raw: str):
    if not raw:
        return None
    text = raw.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    start = text.find("{")
    if start < 0:
        return None
    end = text.rfind("}")
    if end <= start:
        return None
    try:
        return json.loads(text[start:end+1])
    except json.JSONDecodeError:
        return None

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2: FULL-CONTEXT STRESS TEST (survivors only)
# ═══════════════════════════════════════════════════════════════════════════════

FULL_CLASSIFY_CASES = [
    {
        "name": "banter_with_context",
        "prompt_data": {
            "chunks": [{"id": "seg-010", "start": 100, "end": 115, "text": "So I was telling my wife about this and she just laughed. Anyway, it was a funny story. You know how it goes."}],
            "block_summary": "The host shares a personal anecdote before transitioning back to theological discussion about the divine council.",
            "global_outline": "- Banter interlude\n- Divine council theology\n- Psalm 82 analysis",
            "previous_decision": {"id": "seg-009", "label": "keep", "reason": "core theology"},
            "next_chunk_text": "Now back to the divine council, which as we discussed is Yahweh's heavenly court...",
            "kept_claims_so_far": ["Divine council is Yahweh and angels", "Psalm 82 describes judgment"],
            "universe_state": "Sons of God (bene elohim) — divine beings in God's council\nDivine Council — Yahweh's heavenly assembly\nPsalm 82 — God judges among the gods",
        },
        "expected": "drop",
    },
    {
        "name": "concept_already_covered",
        "prompt_data": {
            "chunks": [{"id": "seg-020", "start": 500, "end": 520, "text": "So the divine council, as we've been saying, is where Yahweh sits enthroned with all the angelic beings around him. This is a really important concept we keep coming back to."}],
            "block_summary": "Repetition of divine council concept already covered extensively.",
            "global_outline": "- Divine council introduction\n- Angelic beings and sons of God\n- Mount Sinai as divine mountain\n- Temple as cosmic mountain",
            "previous_decision": {"id": "seg-019", "label": "keep", "reason": "first mention of divine council"},
            "next_chunk_text": "But not everyone agrees with this interpretation of the biblical text of course...",
            "kept_claims_so_far": ["Divine council is Yahweh and angels", "Mount of assembly is sacred space"],
            "universe_state": "Divine Council — Yahweh sits with sons of God\nMount of Assembly — where gods meet",
        },
        "expected": "drop",
    },
    {
        "name": "new_nuance",
        "prompt_data": {
            "chunks": [{"id": "seg-030", "start": 900, "end": 930, "text": "While the divine council is present throughout the OT, one often overlooked aspect is how the prophets functioned as council members themselves, standing in Yahweh's council to receive revelation."}],
            "block_summary": "Discussion of prophetic role in the divine council.",
            "global_outline": "- Prophetic calling as council participation\n- Jeremiah and Micaiah in the council\n- New dimension of divine council theology",
            "previous_decision": {"id": "seg-029", "label": "keep", "reason": "prophet as council member intro"},
            "next_chunk_text": "Let's look at Jeremiah 23 where he describes standing in the council of the Lord...",
            "kept_claims_so_far": ["Divine council is Yahweh and angels"],
            "universe_state": "Divine Council — Yahweh's assembly",
        },
        "expected": "keep",
    },
    {
        "name": "intro_greeting",
        "prompt_data": {
            "chunks": [{"id": "seg-001", "start": 0, "end": 30, "text": "Welcome back to the Lord of Spirits podcast. I'm your host Father Andrew Damick and joining me as always is Father Stephen De Young. Today we're continuing our series on the divine council."}],
            "block_summary": "Opening of the episode with greetings.",
            "global_outline": "- Host introductions\n- Series overview\n- Divine council topic introduction",
            "previous_decision": None,
            "next_chunk_text": "So what exactly do we mean when we talk about the divine council in scripture?",
            "kept_claims_so_far": [],
            "universe_state": "",
        },
        "expected": "drop",
    },
]

FULL_CLASSIFY_PROMPT = """Classify each chunk as keep, drop, or maybe.

DROPS: intros, greetings, ads, banter, filler, repetition (check kept_claims_so_far), concepts already in universe_state (unless new depth).
KEEPS: core theology, scriptural interpretation, historical/ANE facts, clarifying examples, needed transitions.
MAYBE: rare (<5%), only if borderline and dropping breaks coherence.

CRITICAL: If your reason says non-core, MUST label drop. When unsure, drop.

"""


def phase2_classify(model: str) -> dict:
    log(f"\n  {'='*50}")
    log(f"  PHASE 2: FULL-CONTEXT CLASSIFICATION — {model}")
    log(f"  {'='*50}")

    cases = []
    valids = 0
    correct_labels = 0
    echos = 0

    for idx, case in enumerate(FULL_CLASSIFY_CASES):
        payload = json.dumps(case["prompt_data"], ensure_ascii=False, indent=2)
        prompt = FULL_CLASSIFY_PROMPT + payload + '\n\n{"decisions": ['

        log(f"  ── #{idx+1} classify  {case['name']:25s}  ", end="")

        t0 = time.time()
        error = None
        parsed = None
        valid = False
        echoing = False
        label = None

        try:
            raw = generate(prompt=prompt, model=model, timeout=120, temperature=0.1, force_json=True, max_tokens=256)
            elapsed = time.time() - t0
            parsed = extract_json(raw)
            if parsed and isinstance(parsed, dict):
                decs = parsed.get("decisions")
                if decs and len(decs) > 0:
                    label = decs[0].get("label")
                if not label:
                    label = parsed.get("label") or parsed.get("classification")
                valid = label in ("keep", "drop", "maybe")
                echoing = "chunks" in parsed
        except Exception as e:
            elapsed = time.time() - t0
            error = str(e)[:80]

        if valid:
            valids += 1
            correct = label == case["expected"]
            if correct:
                correct_labels += 1

        icon = "✓" if valid and label == case["expected"] else ("~" if valid else "✗")
        echo_tag = " ECHO" if echoing else ""
        err_tag = f" [{error}]" if error else ""
        log(f"{icon}  {elapsed:5.1f}s  label={label}  expected={case['expected']}{echo_tag}{err_tag}")

        if echoing:
            echos += 1

        cases.append({
            "name": case["name"], "valid": valid, "label": label,
            "expected": case["expected"], "echoing": echoing,
            "latency": round(elapsed, 1), "error": error,
        })

    return {
        "cases": cases,
        "passed": correct_labels >= 3 and echos == 0,
        "valid_count": valids,
        "correct_labels": correct_labels,
        "echo_count": echos,
    }


# ── live state (persisted incrementally) ───────────────────────────────────────
PARTIAL_PATH = "benchmark_partial.json"
partial_results = []

def save_partial():
    with open(PARTIAL_PATH, "w", encoding="utf-8") as f:
        json.dump(partial_results, f, indent=2, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1: SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════

SMOKE_CASES = [
    # (task, name, payload_builder)
    ("summarize", "clean_theology", {"block_id": 1, "transcript": "The divine council is a key concept in OT studies. Yahweh presides over the council of divine beings called sons of God. Psalm 82 describes God judging among the gods.", "word_count": 30}),
    ("summarize", "banter_filler", {"block_id": 2, "transcript": "Alright so you know we were talking about that last time and I think it was really interesting anyway moving on. So yeah let's get into it.", "word_count": 25}),
    ("summarize", "historical_context", {"block_id": 3, "transcript": "The Ugaritic texts from Ras Shamra describe El sitting on the mount of assembly surrounded by the sons of El. This is the closest parallel to the biblical divine council.", "word_count": 28}),
    ("classify", "core_theology", {"chunks": [{"id": "seg-001", "text": "The divine council is a foundational concept where Yahweh presides over angelic beings as their king, described in Psalm 82."}]}),
    ("classify", "greeting", {"chunks": [{"id": "seg-002", "text": "Hello everyone, welcome back to the Lord of Spirits podcast. Great to be here with all of you today."}]}),
    ("classify", "ad_break", {"chunks": [{"id": "seg-003", "text": "This episode is brought to you by our patrons on Patreon. Go to patreon.com to support the show."}]}),
]

PROMPT_TEMPLATES = {
    "summarize": lambda p: 'You are a transcript reducer. Output ONLY a JSON object with a "summary" field containing 2-4 very short bullet points (max 15 words each).\n\n' + p + '\n\nReturn valid JSON only. No extra text.',
    "classify": lambda p: 'Classify each chunk as keep, drop, or maybe.\nDROPS: intros, greetings, ads, banter, filler.\nKEEPS: core theology, scriptural interpretation, historical facts.\nMAYBE: rare (<5%).\n\n' + p + '\n\n{"decisions": [',
}

def smoke_test_model(model: str) -> dict:
    log(f"\n  {'='*50}")
    log(f"  PHASE 1: SMOKE TEST — {model}")
    log(f"  {'='*50}")

    vram_start, vram_total = measure_vram()
    log(f"  VRAM: {vram_start}/{vram_total} MB")
    model_results = {"model": model, "vram_start": vram_start, "vram_total": vram_total, "cases": []}
    errors = 0
    echos = 0
    valids = 0

    for idx, (task, name, payload_data) in enumerate(SMOKE_CASES):
        prompt = PROMPT_TEMPLATES[task](json.dumps(payload_data, ensure_ascii=False))

        # heartbeat: what we're about to do
        log(f"  ── #{idx+1} {task:10s} {name:20s}  ", end="")

        t0 = time.time()
        error = None
        parsed = None
        valid = False
        echoing = False
        output_len = 0

        try:
            raw = generate(prompt=prompt, model=model, timeout=120, temperature=0.1, force_json=True, max_tokens=256)
            elapsed = time.time() - t0
            output_len = len(raw)
            parsed = extract_json(raw)
            elapsed = time.time() - t0

            if parsed is None:
                valid = False
            elif isinstance(parsed, dict):
                if task == "summarize":
                    valid = bool(parsed.get("summary"))
                    echoing = "transcript" in parsed or "chunks" in parsed
                elif task == "classify":
                    decs = parsed.get("decisions")
                    label = parsed.get("label") or parsed.get("classification")
                    valid = bool(decs) or label in ("keep", "drop", "maybe")
                    echoing = "chunks" in parsed
            else:
                valid = False
        except Exception as e:
            elapsed = time.time() - t0
            error = str(e)[:80]

        case = {
            "task": task, "name": name, "valid": valid,
            "echoing": echoing, "latency": round(elapsed, 1),
            "output_len": output_len, "error": error,
        }
        model_results["cases"].append(case)
        partial_results.append({"model": model, **case})
        save_partial()

        if valid:
            valids += 1
            icon = "✓"
        elif parsed and not error:
            icon = "~"
        elif error:
            icon = "✗"
            errors += 1
        else:
            icon = "✗"
            errors += 1

        if echoing:
            echos += 1
            echo_tag = " ECHO"
        else:
            echo_tag = ""

        err_tag = f" [{error}]" if error else ""
        log(f"{icon}  {elapsed:5.1f}s  out={output_len}c{echo_tag}{err_tag}")

        # fail-fast: 2 echos → eliminate
        if echos >= 2:
            log(f"  {icon} FAIL-FAST: echoed input {echos}x — stopping {model}")
            break
        # fail-fast: 2 errors → eliminate
        if errors >= 2:
            log(f"  {icon} FAIL-FAST: {errors} errors — stopping {model}")
            break

    vram_end, _ = measure_vram()
    model_results["vram_end"] = vram_end
    model_results["vram_delta"] = vram_end - vram_start
    model_results["passed"] = valids >= (len(model_results["cases"]) * 0.5) and echos == 0 and errors == 0
    model_results["valid_count"] = valids
    model_results["echo_count"] = echos
    model_results["error_count"] = errors

    status = "PASSED" if model_results["passed"] else "FAILED"
    log(f"  ── {status} (valid={valids}, echo={echos}, errors={errors})")
    return model_results


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Unbuffer Python output
    sys.stdout.reconfigure(line_buffering=True)
    os.environ["PYTHONUNBUFFERED"] = "1"

    MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen3:8b"]
    survivors = []
    start_time = time.time()

    log("=" * 60)
    log("  PROGRESSIVE MODEL BENCHMARK")
    log("=" * 60)
    vram_u, vram_t = measure_vram()
    log(f"  GPU VRAM: {vram_u}/{vram_t} MB")
    log(f"  Partial results → {PARTIAL_PATH}")

    # Clear previous partial
    if os.path.exists(PARTIAL_PATH):
        os.remove(PARTIAL_PATH)

    for model in MODELS:
        elapsed = time.time() - start_time
        log(f"\n  ⏱  {elapsed:.0f}s elapsed — testing {model}")
        result = smoke_test_model(model)

        if result["passed"]:
            log(f"\n  ✓ {model} PASSED Phase 1 — advancing to Phase 2")
            p2 = phase2_classify(model)
            result["phase2"] = p2
            if p2["passed"]:
                log(f"\n  ✓ {model} PASSED Phase 2 — strong candidate")
            else:
                log(f"\n  ~ {model} Phase 2: {p2['correct_labels']}/{len(p2['cases'])} correct, {p2['echo_count']} echo — qualified with caveats")
            survivors.append(result)
        else:
            log(f"\n  ✗ {model} ELIMINATED")

        # Check total budget (5 min max per model)
        if time.time() - start_time > 600:
            log("\n  ⏱  Total budget 10m exceeded — stopping")
            break

    log(f"\n{'='*60}")
    log(f"  BENCHMARK COMPLETE ({time.time()-start_time:.0f}s)")
    log(f"  {len(survivors)}/{len(MODELS)} models passed smoke test")
    log(f"{'='*60}")

    for r in survivors:
        s_cases = [c for c in r["cases"] if c["task"] == "summarize"]
        c_cases = [c for c in r["cases"] if c["task"] == "classify"]
        p2 = r.get("phase2", {})
        s_avg = sum(c["latency"] for c in s_cases) / max(1, len(s_cases))
        c_avg = sum(c["latency"] for c in c_cases) / max(1, len(c_cases))
        log(f"\n  ✓ {r['model']}")
        log(f"       VRAM: {r['vram_start']}/{r['vram_total']} MB (delta {r['vram_delta']:+d} MB)")
        log(f"       Phase 1 summarize: {sum(1 for c in s_cases if c['valid'])}/{len(s_cases)} valid  avg {s_avg:.1f}s")
        log(f"       Phase 1 classify:  {sum(1 for c in c_cases if c['valid'])}/{len(c_cases)} valid  avg {c_avg:.1f}s")
        log(f"       Phase 2 full-context: {p2.get('correct_labels',0)}/{len(p2.get('cases',[]))} correct labels  {p2.get('echo_count',0)} echo")

    if survivors:
        # Pick best: Phase 2 accuracy > Phase 1 speed
        best = max(survivors, key=lambda r: (
            r.get("phase2", {}).get("correct_labels", 0) * 1000 +
            sum(1 for c in r["cases"] if c["valid"]) * 100 -
            sum(c["latency"] for c in r["cases"])
        ))
        log(f"\n  ✓ RECOMMENDATION: {best['model']}")
        p2 = best.get("phase2", {})
        log(f"       Reason: reliably structured output, correct labels, low echo, fast enough latency")
        log(f"       Phase 2 full-context: {p2.get('correct_labels',0)}/{len(p2.get('cases',[]))} correct, {p2.get('echo_count',0)} echo")
    else:
        log(f"\n  ✗ No model passed — see partial results in {PARTIAL_PATH}")
