//! WhisprKing entry point.
//!
//! Boot order — each step logs so the failure point is visible:
//!
//!   1. init logger (verbose by default — set `RUST_LOG=warn` for quiet)
//!   2. load config
//!   3. prompt for macOS Accessibility (this is also what makes the app
//!      *appear* in the Privacy & Security list — macOS only lists apps
//!      that have asked for the permission at least once)
//!   4. probe microphone status
//!   5. build the STT engine if the configured model is on disk
//!   6. start the hotkey listener — this is what installs the global key
//!      tap, so this is when macOS may pop the Accessibility dialog
//!   7. boot the eframe main window (blocks until closed)

use std::sync::{Arc, Mutex};

use anyhow::Result;

use whisprking::config::Config;
use whisprking::dictation::DictationHandle;
use whisprking::hotkey::listener::{HotkeyHandler, HotkeyListener};
use whisprking::transcription::engine::{build_transcriber, Transcriber};
use whisprking::transcription::model_manager::ModelManager;
use whisprking::ui::overlay::DictationStatus;
use whisprking::utils::permissions::{
    check_accessibility, check_microphone, prompt_for_accessibility, PermissionStatus,
};

fn main() -> Result<()> {
    // Icon export runs before anything else: it must not load config, ask
    // for Accessibility or open a window, because `just bundle` calls it on
    // a build machine.
    if let Some(code) = handle_icon_args() {
        std::process::exit(code);
    }

    init_logger();
    install_panic_hook();

    log::info!("WhisprKing starting (pid={})", std::process::id());

    let mut config = Config::load_default()?;

    // Older builds routed system audio by creating public Core Audio
    // devices and switching the system default output to one of them. That
    // survived quitting the app, so undo it before doing anything else —
    // a user who crashed mid-meeting is still routed through a device that
    // belongs to a process that is no longer running.
    #[cfg(target_os = "macos")]
    {
        let report = whisprking::audio::legacy::repair(&mut config);
        if let Some(summary) = report.summary() {
            log::warn!("{summary}");
        }
    }

    let config = config;
    let data_dir = Config::resolve_data_dir(&config.data_dir);
    log::info!("config loaded — data_dir={}", data_dir.display());
    log::info!(
        "dictation: hotkey={}, model={}, language={}",
        config.dictation.hotkey,
        config.dictation.model,
        config.dictation.language
    );

    log_permission_state();

    // The OS adds us to the Privacy & Security list the first time we
    // call an Accessibility API; we trigger that here and pop the prompt
    // dialog if we aren't trusted yet.
    let trusted = prompt_for_accessibility();
    if !trusted {
        log::warn!(
            "Accessibility not granted. macOS should now be showing a prompt. \
             After clicking 'Open System Settings' and enabling WhisprKing, \
             QUIT and RESTART this binary — accessibility status is checked \
             at launch."
        );
    }

    let engine: Option<Arc<dyn Transcriber>> = build_engine(&config);
    let config = Arc::new(Mutex::new(config));

    // Shared between the dictation worker and the UI: the worker writes
    // recording / transcribing, the UI paints the on-screen overlay from
    // it. Created here because the worker starts before eframe does.
    let status = DictationStatus::default();

    // Start the dictation pipeline + hotkey listener BEFORE eframe so the
    // rdev event tap is installed immediately; this is what surfaces the
    // Accessibility prompt and what makes key events actually flow.
    let dictation = DictationHandle::spawn(Arc::clone(&config), engine.clone(), status.clone());
    let handler: Arc<dyn HotkeyHandler> = Arc::new(dictation.clone());

    let hotkey_name = {
        let g = config.lock().expect("config");
        g.dictation.hotkey.clone()
    };
    let mut listener = HotkeyListener::new(&hotkey_name, handler)?;
    listener.start();

    log::info!("hotkey listener started — try holding {:?}", hotkey_name);
    log::info!(
        "if no 'rdev event' / 'hotkey activate' lines appear when you press the key, \
         Accessibility is still denied. Grant it and restart."
    );

    // Boot eframe with the engine (cloned for the UI side).
    let ui_engine = engine.map(boxed_clone);
    let ui_config = {
        let g = config.lock().expect("config");
        g.clone()
    };
    if let Err(e) = whisprking::ui::app::run(ui_config, ui_engine, status) {
        log::error!("ui exited with error: {e}");
    }

    Ok(())
}

/// Handle `--write-icns <file>` / `--write-icon-pngs <dir>` and return the
/// exit code, or `None` when the app was started normally.
///
/// The app icon is generated from the same code that draws the mark in the
/// UI (`ui::brand`), so the bundle can never drift from what the app shows.
fn handle_icon_args() -> Option<i32> {
    let args: Vec<String> = std::env::args().collect();
    let value = |flag: &str| -> Option<&String> {
        args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1))
    };

    let result = if let Some(path) = value("--write-icns") {
        whisprking::ui::brand::write_icns(std::path::Path::new(path)).map(|()| vec![path.clone()])
    } else if let Some(dir) = value("--write-icon-pngs") {
        whisprking::ui::brand::write_pngs(std::path::Path::new(dir))
            .map(|paths| paths.iter().map(|p| p.display().to_string()).collect())
    } else {
        return None;
    };

    match result {
        Ok(paths) => {
            for path in paths {
                println!("{path}");
            }
            Some(0)
        }
        Err(e) => {
            eprintln!("icon export failed: {e}");
            Some(1)
        }
    }
}

/// Catch panics from any thread (rdev callback, cpal callback, dictation
/// worker) and surface them in the log. Without this, a panic in a worker
/// thread prints to stderr and the parent `just` recipe sees only the
/// signal — making "as soon as I press the hotkey it crashes" reports
/// impossible to diagnose.
fn install_panic_hook() {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let thread = std::thread::current();
        let name = thread.name().unwrap_or("<unnamed>");
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>");
        let loc = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown loc>".into());
        log::error!("PANIC in thread '{name}' at {loc}: {payload}");
        let bt = std::backtrace::Backtrace::force_capture();
        log::error!("backtrace:\n{bt}");
        prev(info);
    }));
}

/// Default to `info` for the whisprking crate so the diagnostic flow is
/// visible without setting `RUST_LOG`. Set `RUST_LOG=whisprking=trace` to
/// also dump every rdev event.
fn init_logger() {
    let env = env_logger::Env::default().default_filter_or("whisprking=info,info");
    env_logger::Builder::from_env(env)
        .format_timestamp_millis()
        .init();
}

fn log_permission_state() {
    let accessibility = check_accessibility();
    log::info!("accessibility trusted: {}", accessibility);
    match check_microphone() {
        PermissionStatus::Granted => log::info!("microphone: granted"),
        PermissionStatus::Denied => log::warn!("microphone: denied"),
        PermissionStatus::NotDetermined => log::info!("microphone: not determined (will prompt)"),
        PermissionStatus::Unknown => log::info!("microphone: unknown (will prompt at first use)"),
    }
}

fn build_engine(config: &Config) -> Option<Arc<dyn Transcriber>> {
    let manager = ModelManager::new(Config::models_dir(config));
    if !manager.is_downloaded(&config.dictation.model) {
        log::warn!(
            "model {:?} not downloaded — dictation will not transcribe. \
             Use the Settings tab to download it.",
            config.dictation.model
        );
        return None;
    }
    log::info!("building STT engine for model {:?}", config.dictation.model);
    let models_dir = Config::models_dir(config);
    match build_transcriber(
        &config.dictation.model,
        &models_dir,
        &config.dictation.language,
        4,
    ) {
        Ok(engine) => {
            log::info!("STT engine ready");
            Some(Arc::from(engine))
        }
        Err(e) => {
            log::error!("STT engine init failed: {e}");
            None
        }
    }
}

/// The eframe side wants `Box<dyn Transcriber>` but the hotkey controller
/// keeps an `Arc` so the engine is shared. Build a tiny adaptor that
/// forwards through the Arc.
fn boxed_clone(engine: Arc<dyn Transcriber>) -> Box<dyn Transcriber> {
    struct Adaptor(Arc<dyn Transcriber>);
    impl Transcriber for Adaptor {
        fn transcribe(
            &self,
            audio: &[f32],
            sample_rate: u32,
        ) -> Result<String, whisprking::transcription::engine::EngineError> {
            self.0.transcribe(audio, sample_rate)
        }
        // Every method has to be forwarded, not just the ones that look
        // interesting: each one has a default, so a missing forward compiles
        // fine and silently answers for whisper no matter what is behind the
        // Arc. That is how the meeting path would lose carryover and the
        // language pin, and how the post pass would pack a transducer into
        // whisper-sized windows.
        fn transcribe_with_context(
            &self,
            audio: &[f32],
            sample_rate: u32,
            cx: whisprking::transcription::engine::DecodeContext<'_>,
        ) -> Result<String, whisprking::transcription::engine::EngineError> {
            self.0.transcribe_with_context(audio, sample_rate, cx)
        }

        fn transcribe_timed(
            &self,
            audio: &[f32],
            sample_rate: u32,
            cx: whisprking::transcription::engine::DecodeContext<'_>,
        ) -> Result<
            Vec<whisprking::transcription::engine::TimedPiece>,
            whisprking::transcription::engine::EngineError,
        > {
            self.0.transcribe_timed(audio, sample_rate, cx)
        }

        fn max_input_seconds(&self) -> f64 {
            self.0.max_input_seconds()
        }

        fn backend_label(&self) -> &'static str {
            self.0.backend_label()
        }
    }
    Box::new(Adaptor(engine))
}
