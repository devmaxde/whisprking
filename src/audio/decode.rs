//! Decode an on-disk recording to mono `f32` @ 16 kHz for transcription.
//!
//! Live capture already lands as 16 kHz mono via [`Resampler`]. Imported
//! files are the missing case: a meeting recording is usually an `.m4a`
//! (Zoom / QuickTime / voice memo), an `.mp3`, or occasionally a screen-
//! recording video.
//!
//! Two paths:
//! - **Audio** (m4a/aac, mp3, flac, ogg/vorbis, wav) is decoded natively by
//!   [`symphonia`] — pure Rust, no system libraries — and resampled through
//!   the same [`Resampler`] every capture source uses.
//! - **Video** (or any container symphonia can't demux, e.g. mkv/webm/avi)
//!   is handed to the system `ffmpeg` binary when it is installed, which
//!   extracts a 16 kHz mono WAV we then read back. If `ffmpeg` is missing we
//!   reject the file rather than guess.

use std::path::Path;
use std::process::{Command, Stdio};

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use thiserror::Error;

use super::resample::{Resampler, TARGET_RATE};

/// Container extensions that are (usually) video and therefore need
/// `ffmpeg` — symphonia only demuxes ISO-MP4/MOV, so mkv/webm/avi/… route
/// straight to the external tool.
const VIDEO_EXTS: &[&str] = &[
    "mp4", "mov", "m4v", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "mts", "ts", "3gp",
    "ogv",
];

#[derive(Debug, Error)]
pub enum DecodeError {
    #[error("could not read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("no audio track in {0}")]
    NoAudioTrack(String),
    #[error(
        "{0} looks like a video file. Install ffmpeg (e.g. `brew install ffmpeg`) to import \
         video, or export the audio as m4a/mp3/wav first."
    )]
    VideoNeedsFfmpeg(String),
    #[error("ffmpeg failed to extract audio: {0}")]
    Ffmpeg(String),
    #[error("unsupported or unreadable audio in {0}")]
    Unsupported(String),
}

/// Decode any supported recording to mono `f32` at [`TARGET_RATE`] (16 kHz).
///
/// Tries the native decoder first (fast, no external dependency). If that
/// yields nothing usable — an unsupported container, or a video with no
/// symphonia-decodable audio track — it falls back to `ffmpeg`. When
/// `ffmpeg` is also unavailable the file is rejected.
pub fn decode_to_mono_16k(path: &Path) -> Result<Vec<f32>, DecodeError> {
    let display = path.display().to_string();

    if !path.is_file() {
        return Err(DecodeError::Io {
            path: display,
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "not a file"),
        });
    }

    // Native decode covers the common audio formats and audio-only .mp4/.mov.
    match decode_symphonia(path) {
        Ok(samples) if !samples.is_empty() => return Ok(samples),
        Ok(_) => log::info!("decode: native decoder produced no samples for {display}, trying ffmpeg"),
        Err(e) => log::info!("decode: native decode of {display} failed ({e}), trying ffmpeg"),
    }

    // Native path came up empty. ffmpeg is the fallback for video and for
    // anything symphonia can't handle.
    if ffmpeg_available() {
        return decode_via_ffmpeg(path);
    }

    if is_video_ext(path) {
        Err(DecodeError::VideoNeedsFfmpeg(display))
    } else {
        Err(DecodeError::Unsupported(display))
    }
}

fn is_video_ext(path: &Path) -> bool {
    ext_lower(path).is_some_and(|e| VIDEO_EXTS.contains(&e.as_str()))
}

fn ext_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
}

// =====================================================================
// Native decode (symphonia)
// =====================================================================

fn decode_symphonia(path: &Path) -> Result<Vec<f32>, DecodeError> {
    let display = path.display().to_string();
    let file = std::fs::File::open(path).map_err(|source| DecodeError::Io {
        path: display.clone(),
        source,
    })?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = ext_lower(path) {
        hint.with_extension(&ext);
    }

    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| DecodeError::Unsupported(format!("{display}: {e}")))?;
    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| DecodeError::NoAudioTrack(display.clone()))?;
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| DecodeError::Unsupported(format!("{display}: {e}")))?;

    let mut resampler: Option<Resampler> = None;
    let mut out: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            // Clean end of stream, or a mid-stream change we don't follow.
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(DecodeError::Unsupported(format!("{display}: {e}"))),
        };
        if packet.track_id() != track_id {
            continue;
        }
        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = *decoded.spec();
                let r = resampler
                    .get_or_insert_with(|| Resampler::new(spec.rate, spec.channels.count()));
                let mut sbuf = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
                sbuf.copy_interleaved_ref(decoded);
                r.push_interleaved(sbuf.samples(), &mut out);
            }
            // A single corrupt packet is recoverable — skip it.
            Err(SymphoniaError::DecodeError(_)) => continue,
            Err(SymphoniaError::IoError(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break
            }
            Err(e) => return Err(DecodeError::Unsupported(format!("{display}: {e}"))),
        }
    }

    Ok(out)
}

// =====================================================================
// ffmpeg fallback (video + anything symphonia can't demux)
// =====================================================================

fn ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn decode_via_ffmpeg(path: &Path) -> Result<Vec<f32>, DecodeError> {
    let display = path.display().to_string();

    // A named temp .wav ffmpeg writes into and we read back. Kept alive for
    // the duration of the read; dropped (deleted) when this returns.
    let tmp = tempfile::Builder::new()
        .prefix("whisprking-import-")
        .suffix(".wav")
        .tempfile()
        .map_err(|source| DecodeError::Io {
            path: "<tempfile>".into(),
            source,
        })?;

    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-y", "-i"])
        .arg(path)
        // Drop video, force mono 16 kHz PCM WAV — exactly what the engine wants.
        .args(["-vn", "-ac", "1", "-ar"])
        .arg(TARGET_RATE.to_string())
        .args(["-f", "wav"])
        .arg(tmp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| DecodeError::Ffmpeg(format!("could not run ffmpeg: {e}")))?;

    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: String = stderr.lines().rev().take(3).collect::<Vec<_>>().join(" ");
        return Err(DecodeError::Ffmpeg(format!("{display}: {tail}")));
    }

    read_wav_to_mono_16k(tmp.path())
}

/// Read a WAV (any rate / channel count) into mono 16 kHz `f32`. ffmpeg
/// already hands us 16 kHz mono, but routing it through [`Resampler`] keeps
/// one conversion path and copes with an unexpected format.
fn read_wav_to_mono_16k(path: &Path) -> Result<Vec<f32>, DecodeError> {
    let mut reader = hound::WavReader::open(path).map_err(|e| DecodeError::Ffmpeg(e.to_string()))?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;

    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };

    let mut resampler = Resampler::new(spec.sample_rate, channels);
    let mut out = Vec::new();
    resampler.push_interleaved(&interleaved, &mut out);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn video_extensions_are_recognised() {
        assert!(is_video_ext(&PathBuf::from("/x/talk.mp4")));
        assert!(is_video_ext(&PathBuf::from("/x/Screen Recording.MOV")));
        assert!(is_video_ext(&PathBuf::from("/x/call.mkv")));
        assert!(!is_video_ext(&PathBuf::from("/x/memo.m4a")));
        assert!(!is_video_ext(&PathBuf::from("/x/track.mp3")));
        assert!(!is_video_ext(&PathBuf::from("/x/noext")));
    }
}
