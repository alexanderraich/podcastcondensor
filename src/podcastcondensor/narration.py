"""Episode narration — DeepSeek rewrites a digest into spoken prose.

``narrate_episode`` builds one episode's digest (without the trace-back
timestamps — metadata must never leak into spoken audio), sends it to
DeepSeek with ``prompts/narrate_episode.txt``, and writes the resulting
flowing narration to ``output/ep-NNN/narrationNNN.txt``. This is the step
between the summary document and text-to-speech: the digest is structured
(headers, bullets, timestamps) and would be read aloud badly; the narration
is plain prose ready for TTS.
"""

import logging
import os

from podcastcondensor.llm.deepseek import DeepSeekClient, resolve_api_key
from podcastcondensor.summary_doc import (
    _QA_EPISODES,
    _default_output_root,
    _episode_digest,
    _episode_dir_numbers,
    _episode_title,
)
from podcastcondensor.tts import synthesize_narration

logger = logging.getLogger(__name__)

_PROMPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "prompts",
    "narrate_episode.txt",
)


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read().strip()


def narrate_episode(
    output_root: str = "",
    ep_num: int = 1,
    client: DeepSeekClient = None,
    out_path: str = "",
    max_tokens: int = 4000,
) -> str:
    """Write a spoken-narration of one episode; return the narration text.

    Builds the episode digest (no trace-back segments), prompts DeepSeek for
    flowing prose, and writes ``output/ep-NNN/narration.txt``. The client is
    created lazily (API key from env) when not passed, so callers can inject
    a stub in tests. Retries once on failure, matching global_state.py.
    """
    root = output_root or _default_output_root()
    if not client:
        client = DeepSeekClient(api_key=resolve_api_key())

    title = _episode_title(root, ep_num)
    digest = _episode_digest(root, ep_num, title, with_segments=False)
    prompt = _load_prompt() + "\n\n" + digest

    last_error = None
    for _attempt in range(2):
        try:
            narration = client.generate(
                prompt=prompt,
                timeout=600,
                temperature=0.1,
                max_tokens=max_tokens,
            ).strip()
            if not narration:
                raise RuntimeError("empty narration response")
            break
        except Exception as e:  # noqa: BLE001 — mirror global_state retry
            last_error = e
            logger.warning("Narration LLM call failed for ep %d: %s — retrying", ep_num, e)
    else:
        raise RuntimeError(f"Narration failed for ep {ep_num}: {last_error}")

    dest = os.path.abspath(out_path) if out_path else _text_path(root, ep_num)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        f.write(narration + "\n")
    logger.info("Wrote narration for ep %d: %s (%d words)", ep_num, dest,
                len(narration.split()))
    return narration


def _text_path(output_root: str, ep_num: int) -> str:
    return os.path.join(output_root, f"ep-{ep_num:03d}", "narration.txt")


def _existing_narration(output_root: str, ep_num: int) -> str:
    """Load an already-written narration.txt, or "" if none/empty."""
    path = _text_path(output_root, ep_num)
    if os.path.exists(path):
        text = open(path, encoding="utf-8").read().strip()
        if text:
            return text
    return ""


def _mp3_path(output_root: str, ep_num: int) -> str:
    return os.path.join(output_root, f"ep-{ep_num:03d}", "narration.mp3")


def assemble_combined_narration(ep_nums, output_root: str, out_path: str) -> str:
    """Concatenate per-episode narrations into one MP3, triple beep between.

    Order is ``ep_nums`` (episode-ascending for a corpus). Reuses the
    master-cut beep generator. Returns the output path.
    """
    from podcastcondensor.master_cut import _generate_beep_file

    import shutil
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="los-combine-")
    try:
        beep = _generate_beep_file(tmpdir, "beep.mp3", count=3)
        parts = []
        for ep_num in ep_nums:
            mp3 = _mp3_path(output_root, ep_num)
            if not os.path.exists(mp3):
                raise FileNotFoundError(f"Missing narration for ep {ep_num}: {mp3}")
            if parts:  # beep between episodes, not before the first
                parts.append(beep)
            parts.append(mp3)

        list_path = os.path.join(tmpdir, "parts.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", list_path, "-c", "copy", out_path],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {proc.stderr.strip()[:300]}")
        logger.info("Wrote combined narration: %s (%d episodes)", out_path, len(ep_nums))
        return os.path.abspath(out_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_corpus_narrations(
    output_root: str = "",
    start: int = 21,
    end: int = 144,
    combined_out: str = "",
) -> dict:
    """Narrate + TTS every non-Q&A episode in [start, end]; optionally combine.

    Resumable: an episode with ``narration.txt`` is reused (0 LLM calls) and
    one with ``narration.mp3`` is skipped entirely. Q&A episodes are skipped.
    Missing ``global_state.json`` for a non-Q&A episode is an error — the
    batch fails loud at the end (never reported as a clean success with gaps).

    Returns {"episodes": [(ep, "new"|"skipped")], "combined": path-or-""}.
    """
    root = output_root or _default_output_root()
    ep_nums = [n for n in _episode_dir_numbers(root) if start <= n <= end]
    if not ep_nums:
        raise RuntimeError(f"No non-Q&A episodes in range {start}-{end} under {root!r}")

    client = DeepSeekClient(api_key=resolve_api_key())
    errors = []
    results = []
    for ep_num in ep_nums:
        if os.path.exists(_mp3_path(root, ep_num)):
            logger.info("ep %d: already narrated, skipping", ep_num)
            results.append((ep_num, "skipped"))
            continue
        gs = os.path.join(root, f"ep-{ep_num:03d}", "global_state.json")
        if not os.path.exists(gs):
            errors.append((ep_num, f"missing global_state.json"))
            continue
        try:
            text = _existing_narration(root, ep_num)
            if not text:
                text = narrate_episode(output_root=root, ep_num=ep_num, client=client)
            synthesize_narration(text, _mp3_path(root, ep_num))
            results.append((ep_num, "new"))
        except Exception as e:  # noqa: BLE001 — one bad ep must not abort the batch
            logger.error("ep %d failed: %s", ep_num, e)
            errors.append((ep_num, str(e)))

    if errors:
        raise RuntimeError(
            f"Narration batch incomplete: {len(errors)}/{len(ep_nums)} episodes "
            f"failed: " + "; ".join(f"ep {n}: {e}" for n, e in errors[:5])
        )

    combined = ""
    if combined_out:
        combined = assemble_combined_narration(ep_nums, root, combined_out)
    return {"episodes": results, "combined": combined}
