# WhisprKing — Simple Rust Voice Transcriber

## Status

| Module            | State         | Notes                                              |
|-------------------|---------------|----------------------------------------------------|
| `config`          | done          | JSON load/save, deep-merge defaults                |
| `audio::recorder` | done          | cpal mic capture, one-shot + chunked continuous    |
| `audio::system_audio` | done      | BlackHole device detection via cpal                |
| `hotkey`          | done          | Hold-to-talk global hotkey via rdev                |
| `output::smart_paste` | done      | Clipboard stash → Cmd+V via Quartz → restore       |
| `output::transcript_writer` | done | Streaming Markdown writer with `[MM:SS]` segments  |
| `postprocess::llm`| done          | OpenRouter chat + model listing                    |
| `utils::permissions` | done       | macOS Accessibility / Mic checks                   |
| `transcription::model_manager` | done | Catalog, download, extract `.tar.bz2`           |
| `transcription::engine` | partial | `Transcriber` trait; `sherpa-rs` impl behind `--features sherpa` |
| `ui::styles`      | done          | egui dark theme constants                          |
| `ui::tray`        | done          | tray-icon menu with state-driven mic glyph         |
| `ui::overlay`     | done          | Borderless top-most viewport, level meter + spinner|
| `ui::pages::meeting`  | done      | Live transcript view, worker thread, source picker |
| `ui::pages::history`  | done      | List + search + preview + reveal/delete            |
| `ui::pages::settings` | done      | Combos/checkboxes/AI section, persists on change   |
| Tray ↔ overlay ↔ hotkey wiring in `main.rs` | **TODO** | Today `main.rs` only opens the main window; the tray loop and dictation pipeline still need to be hooked in alongside the eframe ticker. |

## Build

```bash
cargo check                  
cargo build --features sherpa --release
```

`--features sherpa` requires the `sherpa-onnx` native library. See the
[`sherpa-rs`](https://crates.io/crates/sherpa-rs) crate for setup.

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

## Design notes
- Errors are `thiserror`-based per module; the binary glues them with
  `anyhow`.
- No global state. Long-lived components (recorder, engine, hotkey) own
  their resources and are passed in explicitly.
- STT is behind a `Transcriber` trait so the rest of the code is testable
  without the native ONNX runtime.
