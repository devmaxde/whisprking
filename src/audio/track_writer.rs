//! Keep each capture track on disk so it can be transcribed again, better.
//!
//! The live meeting pipeline is a one-way street: audio arrives, gets cut into
//! spans, gets decoded, and is gone. That is fine for the running transcript
//! and useless for anything afterwards — a second pass with a bigger model and
//! longer windows needs the audio back.
//!
//! So while a meeting runs, every frame is also appended to a WAV per track.
//! Per *track*, not mixed: keeping "Du" and "Andere" in separate files is what
//! lets the post pass keep speaker attribution without diarization, exactly
//! like the live path does.
//!
//! The files are 16 kHz mono 16-bit — the format every backend wants anyway,
//! about 115 MB per hour per track. Capture already delivers 16 kHz mono
//! `f32`, so this only narrows the samples; no resampling happens here.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use hound::{SampleFormat, WavSpec, WavWriter};
use thiserror::Error;

use super::capture::Track;
use super::resample::TARGET_RATE;

#[derive(Debug, Error)]
pub enum TrackError {
    #[error("could not write {path}: {source}")]
    Write {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("could not read {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: hound::Error,
    },
    #[error("{path} is not 16 kHz mono audio ({rate} Hz, {channels} ch)")]
    WrongFormat {
        path: PathBuf,
        rate: u32,
        channels: u16,
    },
}

fn spec() -> WavSpec {
    WavSpec {
        channels: 1,
        sample_rate: TARGET_RATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    }
}

/// File name for one track of the recording that produced `<stem>.md`.
pub fn track_path(audio_dir: &Path, stem: &str, track: Track) -> PathBuf {
    let suffix = match track {
        Track::Mic => "mic",
        Track::System => "system",
    };
    audio_dir.join(format!("{stem}.{suffix}.wav"))
}

/// Streams one track to a WAV file.
///
/// Write failures are logged once and then swallowed: a full disk must cost
/// the user their post-transcription, never the meeting they are in the middle
/// of recording.
pub struct TrackWriter {
    path: PathBuf,
    writer: Option<WavWriter<BufWriter<File>>>,
    written: u64,
    failed: bool,
}

impl TrackWriter {
    pub fn create(path: PathBuf) -> Result<Self, TrackError> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let writer = WavWriter::create(&path, spec()).map_err(|source| TrackError::Write {
            path: path.clone(),
            source,
        })?;
        Ok(Self {
            path,
            writer: Some(writer),
            written: 0,
            failed: false,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one frame. Values outside `[-1, 1]` are clipped, which is what
    /// any 16-bit capture device would have done to them anyway.
    pub fn push(&mut self, samples: &[f32]) {
        if self.failed {
            return;
        }
        let Some(writer) = self.writer.as_mut() else {
            return;
        };
        for &s in samples {
            let clamped = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            if let Err(e) = writer.write_sample(clamped) {
                log::error!(
                    "track writer: {} failed after {} samples: {e}",
                    self.path.display(),
                    self.written
                );
                self.failed = true;
                return;
            }
            self.written += 1;
        }
    }

    pub fn seconds(&self) -> f64 {
        self.written as f64 / TARGET_RATE as f64
    }

    /// Close the file. Returns its path when it actually holds audio — an
    /// empty track (a meeting where nobody on that side spoke) is deleted
    /// rather than handed to the post pass.
    pub fn finish(mut self) -> Option<PathBuf> {
        if let Some(writer) = self.writer.take() {
            if let Err(e) = writer.finalize() {
                log::error!("track writer: finalize {} failed: {e}", self.path.display());
                self.failed = true;
            }
        }
        if self.failed || self.written == 0 {
            let _ = std::fs::remove_file(&self.path);
            return None;
        }
        log::info!(
            "track writer: {} holds {:.1}s",
            self.path.display(),
            self.seconds()
        );
        Some(self.path.clone())
    }
}

/// Read a track back as the `f32` samples the engines take.
pub fn read_track(path: &Path) -> Result<Vec<f32>, TrackError> {
    let mut reader = hound::WavReader::open(path).map_err(|source| TrackError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != TARGET_RATE {
        return Err(TrackError::WrongFormat {
            path: path.to_path_buf(),
            rate: spec.sample_rate,
            channels: spec.channels,
        });
    }
    let scale = 1.0 / (i16::MAX as f32);
    Ok(reader
        .samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s as f32 * scale)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trips_samples() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.wav");
        let mut w = TrackWriter::create(path.clone()).unwrap();
        let input: Vec<f32> = (0..1_000).map(|i| (i as f32 * 0.01).sin() * 0.8).collect();
        w.push(&input);
        assert!((w.seconds() - 1_000.0 / 16_000.0).abs() < 1e-9);
        assert_eq!(w.finish(), Some(path.clone()));

        let back = read_track(&path).unwrap();
        assert_eq!(back.len(), input.len());
        for (i, (a, b)) in input.iter().zip(&back).enumerate() {
            // 16-bit quantisation is the only difference allowed.
            assert!((a - b).abs() < 1e-4, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn a_silent_track_leaves_no_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        let w = TrackWriter::create(path.clone()).unwrap();
        assert_eq!(w.finish(), None);
        assert!(!path.exists(), "an empty track should not be kept");
    }

    #[test]
    fn out_of_range_samples_clip_instead_of_wrapping() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("hot.wav");
        let mut w = TrackWriter::create(path.clone()).unwrap();
        w.push(&[2.0, -2.0, 0.0]);
        w.finish().unwrap();

        let back = read_track(&path).unwrap();
        assert!(back[0] > 0.99, "positive overshoot must clip high");
        assert!(back[1] < -0.99, "negative overshoot must clip low");
    }

    #[test]
    fn track_paths_are_distinct_and_named_after_the_transcript() {
        let dir = Path::new("/tmp/audio");
        let mic = track_path(dir, "2026-07-26_14-30_meeting", Track::Mic);
        let sys = track_path(dir, "2026-07-26_14-30_meeting", Track::System);
        assert_ne!(mic, sys);
        assert!(mic.ends_with("2026-07-26_14-30_meeting.mic.wav"));
        assert!(sys.ends_with("2026-07-26_14-30_meeting.system.wav"));
    }
}
