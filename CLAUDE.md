# podcastcondensor

Condensing "Lord of Spirits" podcast episodes using DeepSeek LLM.

**Playlist:** https://www.youtube.com/playlist?list=PLZxCUWw2kdo1vAsOOOa3RwzwYvHbybjHR

## Pipeline (4 phases)

Phase 1 downloads the raw SRT. The **global state** phase runs on the full
transcript and produces both the episode outline AND the structured universe
knowledge in a single DeepSeek call. The **raw classifier** receives the raw
SRT file verbatim, the episode outline, and the universe state as context —
one DeepSeek call returns keep/drop per SRT cue entry.

| # | Phase | Artefact | Key file |
|---|-------|----------|----------|
| 1 | **Download** | `source_subtitles.srt` + `.mp3` | `downloader.py` |
| 2 | **Global state** | `global_state.json` (outline + knowledge) | `global_state.py` + `universe_state.py` |
| 3 | **Classify raw** | `decisions.json` (per-entry decisions) | `classify_raw.py` + `prompts/classify_raw.txt` |
| 4 | **Audio cutting** | `condensed_*.mp3` | `audio_strategies.py` + `intervals.py` |

**Resumable:** every phase checks for its artefact before running. If the
artefact exists, the phase is skipped. Interrupted runs pick up where they
left off.

**No timestamp bugs:** Audio cuts use real SRT entry timestamps directly.
The LLM decides keep/drop on raw SRT entries; kept entries go straight
to the interval builder which clusters by time gap and pads.

Phase 2 merges knowledge into the universe state automatically (both in
build-universe and process-playlist modes). No separate extraction phase.

## Universe State (rolling structured knowledge)

Each episode's **global state** phase produces a structured extraction via one
DeepSeek call over the full cleaned transcript. The state accumulates across
episodes:

- `episode_summaries` — 2-3 paragraph narrative per episode
- `entities` — people, places, theological categories (with episode provenance)
- `concepts` — theological concepts and terms
- `claims` — specific doctrinal positions or arguments
- `scriptural_links` — biblical references cited and how they're used
- `glossary` — episode-defined terms (**episode-local**, not universe-wide)

Merged across episodes with dedup by stable ID. The classifier gets the
accumulated universe state as context so it can avoid keeping content
already covered by prior episodes.

**Two modes:**
- `build-universe` — SRT download → clean → **phase 2** (global state) → merge.
  No classification/audio.
- `process-playlist` — full 4-phase pipeline. Phase 2 keeps the state current
  automatically.

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
    decisions.json          # phase 3 — per-entry keep/drop decisions
    condensed_*.mp3         # phase 4 — final audio with beeps
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
SRT → clean → Global state → classify raw → cut audio
               (phase 2)     (3)          (4)
```
