# AGENT.md — podcastcondensor project rules

## Pipeline (4 phases, no more)

1. **Download** — yt-dlp: raw MP3 + raw SRT. No cleaning.
2. **Global state** — single DeepSeek call on the full transcript text. Produces outline + structured knowledge.
3. **Classify raw** — read raw SRT file, inject verbatim into prompt with episode outline + universe state. DeepSeek returns `kept_ranges` and `dropped_ranges` by cue number. Expand to per-entry `{id, label, reason}` for the audio cutter.
4. **Audio cutting** — reverted to remote. Zero changes allowed. Build intervals from kept entries, ffmpeg cut.

**Never add phases, never reintroduce segmentation/punctuation/sentence-mapping.**

## Rules

- **No preprocessing.** Read the raw SRT file and inject it directly. No reformatting, no normalization, no counting, no utterance merging, no cue renumbering unless explicitly asked.
- **Audio cutting code is frozen.** `intervals.py` and `audio_strategies.py` are at zero diff vs origin/main. They stay that way.
- **Dead code is deleted.** `classifier.py`, `segmentation/deepseek.py`, `segmentation/schemas.py`, `segmentation/validation.py` are gone. Do not recreate them.
- **No memory files.** Don't create .claude/projects memory entries for this repo.
- **One DeepSeek call per phase.** Phase 2 and Phase 3 are each a single LLM call. No batching, no chunking, no retry loops beyond one retry.
- **Keep it simple.** The pipeline should fit in your head. If you're adding layers, you're doing it wrong.
- **Don't add what wasn't asked for.** Every time you preprocess, reformat, or "helpfully" transform something the user didn't ask about, you waste time.
- **Don't delete working output.** Completed runs produced valuable artifacts. Don't wipe them for "fresh tests."

## Relevant files

| File | Role |
|------|------|
| `src/podcastcondensor/pipeline.py` | Orchestrator — 4 phases, resumable |
| `src/podcastcondensor/classify_raw.py` | Phase 3: sends raw SRT to DeepSeek, parses JSON |
| `src/podcastcondensor/intervals.py` | Phase 4: build audio intervals (frozen, from remote) |
| `src/podcastcondensor/audio_strategies.py` | Phase 4: ffmpeg cutting (frozen, from remote) |
| `src/podcastcondensor/config.py` | Config with cluster_gap, padding, audio params |
| `src/podcastcondensor/global_state.py` | Phase 2: outline + structured knowledge |
| `src/podcastcondensor/universe_state.py` | Cross-episode knowledge accumulation |
| `prompts/classify_raw.txt` | Prompt for Phase 3 |
| `prompts/global_state.txt` | Prompt for Phase 2 |
| `CLAUDE.md` | Pipeline docs, open points |

## Crash recovery / resume

The pipeline is fully resumable — every phase checks for its artefact before running. If a phase crashes, just re-run the same command.

**Diagnostic artifacts after a crash:**

| File | What it captures |
|------|-----------------|
| `output/ep-NNN/_transcribe_diag.log` | Every `logger.info/warning/error` from the transcribe module + watchdog heartbeats every 30s + exception tracebacks. All fsynced immediately — survives WSL OOM kill. |

**Transcribe crash troubleshooting:** If the diag log ends at `CALLING model.transcribe()` with no watchdog tick, the crash is in the first 30s of the faster-whisper call (likely GPU OOM or driver panic). Lower `whisper_beam_size` (default 1) and set `whisper_vad_filter=False` in `config.py`. If watchdog ticks appear, the model is running but eventually runs out of memory or time.

**Never hook fd 2.** Do not use `os.dup2` to redirect stderr — it can leave the terminal in a broken state if the process crashes mid-redirect, killing the parent Claude session.

## When you're stuck

Do not iterate blindly. Compile a diagnostic with the actual data, the relevant code, and what's needed — hand it to the user.
