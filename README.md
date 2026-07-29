# WhisprKing — Simple Rust Voice Transcriber

## Status

| Module            | State         | Notes                                              |
|-------------------|---------------|----------------------------------------------------|
| `config`          | done          | JSON load/save, deep-merge defaults                |
| `audio::recorder` | done          | cpal mic capture, one-shot + streaming drain       |
| `audio::resample` | done          | Anti-aliased downmix + resample to 16 kHz mono     |
| `audio::tap`      | done          | System audio via Core Audio process tap (14.4+)   |
| `audio::capture`  | done          | Mic + system as two speaker-attributed tracks     |
| `audio::legacy`   | done          | Undoes the old BlackHole routing at startup       |
| `audio::system_audio` | done      | Capability probe for system-audio capture         |
| `hotkey`          | done          | Hold-to-talk global hotkey via rdev                |
| `output::smart_paste` | done      | Clipboard stash → Cmd+V via Quartz → restore       |
| `output::transcript_writer` | done | Streaming Markdown writer with `[MM:SS]` segments  |
| `postprocess::llm`| done          | OpenRouter chat + model listing                    |
| `utils::permissions` | done       | macOS Accessibility / Mic checks                   |
| `audio::vad`      | done          | Silero VAD segmentation, fixed-window fallback     |
| `audio::import`   | done          | Offline file → transcript + LLM summary            |
| `output::transcript_doc` | done   | Recording = transcript + derived documents; legacy split |
| `postprocess::refine` | done      | Runs a preset over a transcript into its own file  |
| `transcription::model_manager` | done | Catalog, download, extract `.tar.bz2`; Silero VAD asset |
| `transcription::context` | done   | Prompt carryover + degenerate-decode guard         |
| `transcription::engine` | partial | `Transcriber` trait; `sherpa-rs` impl behind `--features sherpa` |
| `ui::theme`       | done          | Design tokens, type scale, variable-weight fonts   |
| `ui::brand`       | done          | The crowned-mic mark; app `.icns`, menu bar, rail  |
| `ui::icons`       | done          | Vector icon set drawn with the painter             |
| `ui::widgets`     | done          | Buttons, cards, segmented control, toggles, banners |
| `ui::tray`        | done          | tray-icon menu with state-driven mic glyph         |
| `ui::overlay`     | done          | Borderless top-most viewport, level meter + spinner|
| `ui::pages::meeting`  | done      | Transport, source picker, live dual-track transcript, import |
| `ui::pages::history`  | done      | Recording list + per-document tabs, copy, KI actions |
| `ui::pages::settings` | done      | Cards of setting rows, model download, AI section  |
| Tray ↔ overlay ↔ hotkey wiring in `main.rs` | done | `DictationStatus` is the one shared source of truth: the dictation worker writes it, the app paints the overlay and recolors the tray from it. |

## Build

```bash
cargo check                  
cargo build --features sherpa --release
```

`--features sherpa` requires the `sherpa-onnx` native library. See the
[`sherpa-rs`](https://crates.io/crates/sherpa-rs) crate for setup.

## Handing it to someone else

```bash
just bundle         # → …/bundle/WhisprKing.app + …/bundle/WhisprKing-0.1.0.dmg
just install        # the same, and copies the .app into /Applications
```

There is no separate disk-image command: anything that produces a `.app`
produces the installer next to it, so the thing you hand out can never lag
behind the thing you tested. `just run` is untouched and still goes straight
from `cargo build` to the binary, so the dev loop pays nothing for this.

The disk image opens on a window with the app on the left and an alias to
`/Applications` on the right: drag one onto the other. The layout is recorded
by mounting a writable image, arranging the window through the Finder and only
then compressing it — if macOS denies the automation prompt, the build says so
and still produces a working disk image with default icon positions.

The bundle inside is ad-hoc signed and **not notarized**, so a downloaded copy
is blocked on first launch. Right-click → *Open*, or:

```bash
xattr -dr com.apple.quarantine /Applications/WhisprKing.app
```

## Running & first-launch permissions

```bash
cargo run --release           # plus --features sherpa for actual STT
```

On macOS the binary needs **Accessibility** so it can (a) install a
global keyboard event tap (rdev / hold-to-talk) and (b) synthesize Cmd+V.
On first launch:

1. WhisprKing calls `AXIsProcessTrustedWithOptions(prompt=true)` — macOS
   pops a dialog and adds the binary to *System Settings → Privacy &
   Security → Accessibility*.
2. Flip the switch ON for the binary.
3. **Quit and relaunch.** The trust state is read once at process start.

The log on launch makes it obvious which step you are at:

```
INFO  whisprking: WhisprKing starting (pid=…)
INFO  whisprking: accessibility trusted: false
INFO  whisprking::utils::permissions: AXIsProcessTrustedWithOptions(prompt=true) → false
INFO  whisprking::hotkey::listener: hotkey listener spawning — target key: MetaRight
INFO  whisprking::hotkey::listener: hotkey thread alive, calling rdev::listen()
```

When Accessibility is granted, key events flow and you see:

```
INFO  whisprking::hotkey::listener: hotkey activate
INFO  whisprking::dictation: dictation: starting capture
INFO  whisprking::hotkey::listener: hotkey deactivate
INFO  whisprking::dictation: dictation: captured … samples …
```

If you press the configured hotkey and nothing logs at all, Accessibility
is still denied. Set `RUST_LOG=whisprking=trace` to also dump every rdev
event for further debugging.

## The dictation overlay

While you hold the hotkey (right ⌘ by default) a small pill appears at the
bottom centre of the screen: a live level meter and *Ich höre zu …* while you
speak, a spinner and *Wird transkribiert …* from the moment you let go until
the text is pasted. The menu bar glyph recolors along with it. Neither depends
on the main window being open — the usual case is that it is not.

Two things make that work, and both are load-bearing:

- The overlay is declared from `App::logic`, not `App::ui`. eframe skips `ui`
  entirely for a hidden viewport, so an overlay drawn there would only ever
  appear for users who left the main window open.
- Writes to `DictationStatus` (from the dictation worker thread) call
  `request_repaint`. A hidden window's event loop is otherwise idle, so
  without the wake the state would change and nothing would draw it.

The overlay window is created with `active: false` and mouse passthrough: it
must never become the key window, or the Cmd+V synthesized after transcribing
would land in the overlay instead of the app you were typing into. It is also
hidden *before* that paste is sent.

## Meeting audio

WhisprKing captures the two sides of a call as **separate tracks**:

- **You** — your microphone, opened read-only through cpal.
- **Others** — everything the machine is playing, captured with a
  [Core Audio process tap](https://developer.apple.com/documentation/coreaudio)
  (macOS 14.4+).

Both are transcribed independently, so the transcript is speaker-attributed
without running diarization.

### Transcription quality

Each track is cut into transcription units on **speech boundaries**, not on a
clock, using a Silero VAD (`sherpa` builds; the ~2 MB model is fetched once on
first use). Fixed 10 s windows used to slice words in half several times a
minute, which is what made meeting transcripts poor — a transducer fed a
fragment starting mid-syllable emits garbage. Builds without the `sherpa`
feature fall back to fixed windows.

On top of that, whisper decodes are primed with the tail of the previous
segment on the same track (`initial_prompt`), retried at rising temperature
when a decode looks degenerate, and pinned to a language.

### The second pass

Everything above happens under one constraint: a segment cannot be transcribed
before it has been spoken. That costs accuracy twice — every span is decoded
with no idea of the sentence it sits in, and the model has to keep up in real
time, which rules out the accurate ones.

Once the meeting stops, neither applies. *Einstellungen → Nachbearbeitung*
(on by default) runs the recording again:

- **Both tracks are kept as 16 kHz mono WAVs** while recording
  (`<audio_dir>/<stem>.mic.wav`, `<stem>.system.wav`, ~115 MB per hour per
  track). Per track, not mixed, so the second pass keeps speaker attribution
  for free just like the live one.
- **Spans are glued back into the largest window each backend can use** and
  cut *only* where the VAD found silence — 28 s for whisper's mel window,
  120 s for a transducer. Nothing is ever decoded mid-word, and a sentence
  spread over two utterances is decoded as one. The silence between glued
  spans is restored, capped at 0.6 s, and each window remembers where its
  source spans landed so timestamps still point at when things were said.
- **Every selected model runs at the same time**, each in its own thread with
  its own engine — whisper on Metal and Parakeet on CPU ONNX genuinely run in
  parallel rather than queueing. The default pair is `whisper-large-v3` and
  `parakeet-v3`; anything not downloaded is skipped with a note rather than
  substituted silently.
- **The variants are reconciled by the LLM**, in batches aligned on the
  recording clock (not by line position — the models disagree about how many
  lines a stretch of speech is). A batch that comes back empty or a fraction
  of its input keeps the variant instead: a bad merge must not silently delete
  minutes of transcript. With no AI provider the best single variant becomes
  the result, and the pass says so.

Results are two more sibling documents, `<stem>.post.md` and
`<stem>.post-variants.md`; the live transcript is never touched. While the
audio is kept, *Verlauf → Nachbearbeitet → Nachbearbeitung starten* re-runs the
whole thing with whatever models are selected now.

The merge prompt is `src/postprocess/prompts/reconcile.txt` and is editable
like any other (*Einstellungen → Prompts → Modelle zusammenführen*).

### Models

*Einstellungen → Modell* lists them smallest first:

| id | file | size | notes |
|----|------|------|-------|
| `whisper-tiny` / `whisper-base` / `whisper-small` | `ggml-*.bin` | 75 / 142 / 466 MB | fallbacks |
| `parakeet-v3` | sherpa tar.bz2 | 464 MB | needs `--features sherpa` |
| `whisper-turbo` (default) | `ggml-large-v3-turbo-q5_0.bin` | 574 MB | 5-bit quantized |
| `whisper-turbo-f16` | `ggml-large-v3-turbo.bin` | 1.6 GB | same model, unquantized |
| `whisper-large-v3` | `ggml-large-v3.bin` | 3.1 GB | full 32-layer decoder |

The default is the quantized turbo because it is the one that fits a
hold-to-talk dictation loop: it loads in a fraction of the time and holds a
third of the memory. Quantization is not free — q5_0 rounds the weights to five
bits and pays for it mostly on proper nouns, numbers and rare words. On Apple
Silicon the f16 files decode through Metal at no meaningful penalty per second
of audio, so the price of `whisper-turbo-f16` is disk and RAM, not speed; pick
it if accuracy matters more than a 1.6 GB download.

`whisper-large-v3` is a different trade-off again. Turbo *is* large-v3 with the
decoder cut from 32 layers to 4, which is where its speed comes from; the full
model decodes several times slower and is the most accurate whisper there is.
Worth it for imported recordings, painful for dictation.

Switching models does not delete the old one — each lives in its own directory
under the models dir, so going back is instant.

**Meeting language** is its own setting (Settings → *Meeting-Sprache*), because
a meeting is decoded segment by segment and `Auto` re-detects on every one —
a single English loanword in a German sentence can flip whisper into
*translating* the rest of the call. Leave it on *Wie Diktat* to follow the
dictation language, or pin one explicitly.

**Nothing about your audio setup is changed.** The default output device is
never switched, no virtual driver is required, and the tap lives inside a
*private* aggregate device that no other application can see. Leave your
meeting app pointed at your real microphone.

> **Upgrading from a BlackHole build?** Older versions created a public
> Multi-Output Device, made it the system default output, and published an
> aggregate "WhisprKing Meeting Input" that could be selected as a
> microphone — where it fed callers system audio on channel 1 instead of
> your voice. On first launch the new build tears all of that down and
> restores your previous output device. BlackHole itself can be uninstalled.

System audio capture needs the **Audio Recording** permission; macOS prompts
on first use. Below macOS 14.4 the "Others" track is unavailable and
WhisprKing records your microphone only.

## Transcripts on disk

One recording is one transcript file plus one file per derived document:

```text
2026-07-26_14-30_meeting.md                ← transcript, never rewritten
2026-07-26_14-30_meeting.post.md           ← second pass, variants reconciled
2026-07-26_14-30_meeting.post-variants.md  ← each model's second pass, side by side
2026-07-26_14-30_meeting.cleanup.md        ← LLM cleanup
2026-07-26_14-30_meeting.summary.md        ← summary
2026-07-26_14-30_meeting.action-items.md   ← action items
2026-07-26_14-30_meeting.meta.json         ← which of the above exist, and when
```

The kept audio lives under `<data_dir>/audio/` rather than next to the
transcript, and is removed with the recording when you delete it.

Older builds appended each LLM result to the end of the transcript instead,
which made the corrected text impossible to copy on its own and fed the
previous result back into the model on every re-run. Files in that shape are
split into the layout above the first time the Verlauf tab lists them —
nothing to do by hand. The `Kopieren` button in each tab copies exactly that
document's text, without the Markdown header.

## Prompts

Every preset ships with a system prompt (`src/postprocess/prompts/*.txt`) and
every one of them is editable in *Einstellungen → Prompts*. An edit is stored
per preset in `ai_postprocess.prompts` and only overrides that preset; an empty
entry means "use the shipped text", which is also what *Auf Standard
zurücksetzen* restores. The prompt is sent as the system message and the
transcript as the user message — nothing else is added.

## When a KI run fails

Failures are reported with the facts that explain them, not just a message:
which action, provider and model, how long the transcript was in characters
and estimated tokens, the HTTP status and error code, and the provider's answer
verbatim. One button copies the whole report.

The frequent cause for long recordings was a flat 120 s client timeout — the
model was still writing when WhisprKing hung up, and a client-side timeout is
indistinguishable from any other transport error in a banner. The limit now
scales with the amount of text (90 s plus a minute per 10k characters, capped
at 20 minutes) and a timeout says so. Two more failure modes that used to be
invisible are named now: an OpenAI-style `error` object returned with HTTP 200
(what a context-length overflow usually looks like), and a decode that stopped
at `finish_reason: length`, which would otherwise have been saved as a
silently truncated document.

## UI

The look lives in `ui/theme.rs` (color, spacing, radius and type tokens),
`ui/icons.rs` (vector icons) and `ui/widgets.rs` (buttons, cards, segmented
controls, toggles, banners, list rows). Pages compose those; nothing else
hardcodes a color or a pixel value, so a restyle is a one-file change.

Weights are real: egui's bundled font ships a single light weight, so the app
loads the system UI font (SF Pro on macOS — a variable font) three times at
different `wght` coordinates and registers them as separate families. If no
system font can be read, egui's default is used and the app still runs, just
flatter. `WHISPRKING_FONT` / `WHISPRKING_FONT_MONO` point the loader at an
explicit file.

## The icon

The mark is a microphone wearing a crown, defined once in `ui/brand.rs` as
signed distance fields and rasterized on demand — so the app bundle icon, the
menu-bar icon and the mark in the navigation rail are literally the same
artwork, and the repo carries no image files.

```bash
just icons          # write PNGs (16 … 1024 px + menu bar) to target/icons
just bundle         # writes Contents/Resources/AppIcon.icns as part of the .app
```

`just bundle` produces the `.icns` by calling the binary itself
(`whisprking --write-icns <path>`), which assembles the archive by hand — so
it works on any machine, not only where `iconutil` exists.

Two drawings of the mark exist. Above 48 px it has a cradle, a stem and a base
bar; below that those land on the same pixel row and turn to mush, so a
compact drawing with thicker strokes and fewer parts takes over. That is what
the 22 px menu-bar icon uses, recolored per state (idle / recording /
transcribing / meeting).

> The Dock tile follows the window: WhisprKing runs as a normal foreground
> app while the main window is open (Dock tile, Cmd+Tab, app menu, and a
> window a window manager can actually place), and switches to the accessory
> policy — menu bar only — when the window is closed and just the tray and
> the hotkey stay alive. That switch is `src/ui/activation.rs`; the bundle
> deliberately does **not** set `LSUIElement`, because pinning it there makes
> the app an agent whose window no Dock, Cmd+Tab or window manager can see.

## Design notes
- Errors are `thiserror`-based per module; the binary glues them with
  `anyhow`.
- No global state. Long-lived components (recorder, engine, hotkey) own
  their resources and are passed in explicitly.
- Anything that mutates system-wide audio state is a bug. The capture path
  never writes `meeting.input_device` and never calls
  `set_default_output_by_uid`; both belong to the user.
- STT is behind a `Transcriber` trait so the rest of the code is testable
  without the native ONNX runtime.
