# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Pipeline (6 phases)

Once the SRT is downloaded and cleaned programmatically, the **global state** phase runs on the full transcript and produces both the episode outline AND the structured universe knowledge in a single DeepSeek call. Segmentation then groups entries, and the classifier uses the universe state as context.

**Critical:** Phase 3 has an intermediate artefact — the DeepSeek punctuation call (~3 min for 116k chars) is checkpointed to `punctuated_text.json` so a failure in sentence-to-entry mapping or the segmentation call does not repeat the expense.

| # | Phase | Artefact | Key file |
|---|-------|----------|----------|
| 1 | **Download** | `source_subtitles.srt` + `.mp3` | `downloader.py` |
| 2 | **Global state** | `global_state.json` (outline + knowledge) | `global_state.py` + `universe_state.py` |
| 3a | **Punctuation** (checkpoint) | `punctuated_text.json` | `segmentation/deepseek.py` |
| 3b | **Segmentation** | `segments.json` | `segmentation/deepseek.py` |
| 4 | **Classification** | `decisions.json` | `classifier.py` + `prompts/classify_chunks_global.txt` |
| 5 | **Finalize decisions** | `decisions_final.json` | `classifier.py` |
| 6 | **Audio cutting** | `condensed_*.mp3` | `audio_strategies.py` |

**Resumable:** every phase checks for its artefact before running. If the artefact exists, the phase is skipped. Interrupted runs pick up where they left off.

### Timestamp recovery (no fake timestamps)

When auto-captions have no punctuation (the common case), phase 3 works in three steps:

1. **Punctuate** — DeepSeek adds sentence-ending punctuation to the deduped transcript (~3 min, checkpointed).
2. **Map** — each punctuated sentence is matched back to the original SRT entries via forward-scanning word-overlap scoring. This recovers real timestamps from the SRT. **Never falls back to proportional/interpolated timestamps**, which would produce audio cut at wrong positions.
3. **Segment** — DeepSeek groups sentences into topical segments.

If the mapping stage detects >5 sentences with zero word overlap, a `ValueError` is raised — this means the punctuated output has seriously diverged from the source transcript.

Phase 2 merges knowledge into the universe state automatically (both in build-universe and process-playlist modes). No separate extraction phase.

## Universe State (rolling structured knowledge)

Each episode's **global state** phase produces a structured extraction via one DeepSeek call over the full cleaned transcript. The state accumulates across episodes:

- `episode_summaries` — 2-3 paragraph narrative per episode
- `entities` — people, places, theological categories (with episode provenance)
- `concepts` — theological concepts and terms
- `claims` — specific doctrinal positions or arguments
- `scriptural_links` — biblical references cited and how they're used
- `glossary` — episode-defined terms (**episode-local**, not universe-wide)

Merged across episodes with dedup by stable ID. The classifier gets the accumulated universe state as context so it knows what has already been covered.

**Two modes:**
- `build-universe` — SRT download → clean → **phase 2** (global state) → merge. No segmentation/audio.
- `process-playlist` — full 6-phase pipeline. Phase 2 keeps the state current automatically.

## Required

- DeepSeek API key in `ANTHROPIC_AUTH_TOKEN` or `DEEPSEEK_API_KEY`
- ffmpeg

## Commands

```bash
# Build universe state from episodes 1-21 (SRT + one DeepSeek call per episode)
python3 -m podcastcondensor build-universe [PLAYLIST_URL] --start 1 --end 21

# Process episodes 22+ with universe state (full pipeline)
python3 -m podcastcondensor process-playlist [PLAYLIST_URL] \
  --state-file output/universe_state.json --start 22

# Doctor check
python3 -m podcastcondensor doctor --check
```

## Output

```
output/
  universe_state.json       # accumulated universe knowledge across episodes
  ep-NNN/
    source_subtitles.srt    # phase 1 — raw downloaded SRT
    global_state.json       # phase 2 — outline + structured knowledge
    segments.json           # phase 3 — segmented transcript
    decisions.json          # phase 4 — raw classifier output
    decisions_final.json    # phase 5 — finalised decisions after cleanup
    condensed_*.mp3         # phase 6 — final audio with beeps
    stats.json              # post-run stats
```

## Architecture

```
                        build-universe              process-playlist
                        ─────────────              ────────────────
SRT → clean ──→ Global state ──→ merge into state
                  (phase 2)         │
                        (universe state used as context during classification)
                                    │
SRT → clean → Global state → segment → classify → finalize → cut audio
               (phase 2)     (3)       (4)         (5)        (6)
```

