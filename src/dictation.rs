//! Hold-to-talk dictation pipeline.
//!
//! cpal's audio `Stream` is `!Send` on macOS, so the recorder is pinned
//! to a dedicated worker thread. The hotkey listener (on its own rdev
//! thread) talks to the worker through an `mpsc` channel; the worker is
//! the only place that touches the recorder, the engine, and `smart_paste`.
//!
//! [`DictationHandle`] is just the sending end of that channel — it is
//! `Send + Sync`, so it satisfies the `HotkeyHandler` bounds without
//! resorting to `unsafe impl Send`.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::audio::recorder::{AudioRecorder, SAMPLE_RATE};
use crate::config::Config;
use crate::hotkey::listener::HotkeyHandler;
use crate::output::smart_paste::smart_paste;
use crate::postprocess::llm::{make_provider, resolve_system_prompt, LlmConfig};
use crate::transcription::engine::Transcriber;

#[derive(Debug)]
enum Cmd {
    Press,
    Release,
    Shutdown,
}

/// Handle for the dictation worker. Cheap to clone (it wraps a channel
/// sender + the join handle behind an `Arc`).
#[derive(Clone)]
pub struct DictationHandle {
    tx: Sender<Cmd>,
    worker: Arc<Mutex<Option<JoinHandle<()>>>>,
}

impl DictationHandle {
    pub fn spawn(config: Arc<Mutex<Config>>, engine: Option<Arc<dyn Transcriber>>) -> Self {
        let (tx, rx) = mpsc::channel::<Cmd>();
        let join = thread::Builder::new()
            .name("whisprking-dictation".into())
            .spawn(move || {
                log::info!("dictation worker thread alive");
                let mut state = WorkerState::new(engine, config);
                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        Cmd::Press => state.on_press(),
                        Cmd::Release => state.on_release(),
                        Cmd::Shutdown => break,
                    }
                }
                log::info!("dictation worker thread exiting");
            })
            .expect("spawn dictation worker");
        Self {
            tx,
            worker: Arc::new(Mutex::new(Some(join))),
        }
    }

    pub fn shutdown(&self) {
        let _ = self.tx.send(Cmd::Shutdown);
        if let Some(handle) = self.worker.lock().expect("worker handle").take() {
            let _ = handle.join();
        }
    }
}

impl HotkeyHandler for DictationHandle {
    fn on_activate(&self) {
        log::debug!("dictation: on_activate → Cmd::Press");
        if let Err(e) = self.tx.send(Cmd::Press) {
            log::error!("dictation: send Press failed: {e}");
        }
    }
    fn on_deactivate(&self) {
        log::debug!("dictation: on_deactivate → Cmd::Release");
        if let Err(e) = self.tx.send(Cmd::Release) {
            log::error!("dictation: send Release failed: {e}");
        }
    }
}

struct WorkerState {
    recorder: AudioRecorder,
    engine: Option<Arc<dyn Transcriber>>,
    config: Arc<Mutex<Config>>,
    recording: bool,
}

impl WorkerState {
    fn new(engine: Option<Arc<dyn Transcriber>>, config: Arc<Mutex<Config>>) -> Self {
        Self {
            recorder: AudioRecorder::new(),
            engine,
            config,
            recording: false,
        }
    }

    fn on_press(&mut self) {
        if self.recording {
            log::debug!("dictation: press while already recording — ignoring");
            return;
        }
        log::info!("dictation: starting capture");
        match self.recorder.start() {
            Ok(()) => self.recording = true,
            Err(e) => log::error!("dictation: recorder.start() failed: {e}"),
        }
    }

    fn on_release(&mut self) {
        if !self.recording {
            log::debug!("dictation: release with no active recording");
            return;
        }
        self.recording = false;
        let audio = self.recorder.stop();
        log::info!(
            "dictation: captured {} samples ({:.2}s)",
            audio.len(),
            audio.len() as f32 / SAMPLE_RATE as f32
        );

        let Some(engine) = self.engine.clone() else {
            log::warn!("dictation: no engine loaded — discarding audio");
            return;
        };

        let text = match engine.transcribe(&audio, SAMPLE_RATE) {
            Ok(t) => t.trim().to_string(),
            Err(e) => {
                log::error!("dictation: transcription failed: {e}");
                return;
            }
        };
        if text.is_empty() {
            log::info!("dictation: empty transcript");
            return;
        }
        log::info!("dictation: transcribed {} chars", text.len());

        let final_text = self.maybe_postprocess(&text);
        match smart_paste(&final_text) {
            Ok(true) => log::info!("dictation: pasted"),
            Ok(false) => log::debug!("dictation: nothing to paste"),
            Err(e) => log::error!("dictation: paste failed: {e}"),
        }
    }

    fn maybe_postprocess(&self, text: &str) -> String {
        let (autorun, llm_cfg, preset, custom_prompt, overrides) = {
            let guard = self.config.lock().expect("config");
            let ai = &guard.ai_postprocess;
            (
                ai.dictation_autorun,
                LlmConfig::from_ai_section(ai),
                ai.dictation_preset.clone(),
                ai.custom_prompt.clone(),
                ai.prompts
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect::<std::collections::HashMap<_, _>>(),
            )
        };
        if !autorun {
            return text.to_string();
        }
        let Some(provider) = make_provider(&llm_cfg) else {
            log::info!("dictation: autorun enabled but no provider available");
            return text.to_string();
        };
        let system = resolve_system_prompt(&preset, &custom_prompt, &overrides);
        match provider.run(&system, text) {
            Ok(out) => {
                log::info!(
                    "dictation: LLM cleaned ({} → {} chars)",
                    text.len(),
                    out.len()
                );
                out
            }
            Err(e) => {
                log::warn!("dictation: LLM postprocess failed: {e}");
                text.to_string()
            }
        }
    }
}
