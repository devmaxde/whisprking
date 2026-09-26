//! Stream meeting segments to a Markdown file with `[MM:SS]` timestamps.
//!
//! Filename is `<YYYY-MM-DD_HH-MM>_<slug>.md` under `transcripts_dir`. The
//! header reserves a duration placeholder that `finalize()` rewrites once
//! the meeting ends.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Local};
use regex::Regex;
use thiserror::Error;

const DURATION_PLACEHOLDER: &str = "_(in progress …)_";

#[derive(Debug, Error)]
pub enum WriterError {
    #[error("transcript_writer: start_new() not called")]
    NotStarted,
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

pub struct TranscriptWriter {
    transcripts_dir: PathBuf,
    path: Option<PathBuf>,
    started_at: Option<DateTime<Local>>,
}

impl TranscriptWriter {
    pub fn new(transcripts_dir: impl Into<PathBuf>) -> Result<Self, WriterError> {
        let transcripts_dir = transcripts_dir.into();
        std::fs::create_dir_all(&transcripts_dir).map_err(|source| WriterError::Io {
            path: transcripts_dir.clone(),
            source,
        })?;
        Ok(Self {
            transcripts_dir,
            path: None,
            started_at: None,
        })
    }

    /// Create a new transcript file and write the header. Returns its path.
    pub fn start_new(&mut self, title: Option<&str>) -> Result<&Path, WriterError> {
        let now = Local::now();
        let slug = title.map(slugify).unwrap_or_else(|| "meeting".to_string());
        let filename = format!("{}_{}.md", now.format("%Y-%m-%d_%H-%M"), slug);
        let path = self.transcripts_dir.join(filename);

        let header = format!(
            "# {title}\n\n\
             - **Start:** {start}\n\
             - **Duration:** {placeholder}\n\n\
             ---\n\n",
            title = title.unwrap_or("Meeting"),
            start = now.format("%Y-%m-%d %H:%M"),
            placeholder = DURATION_PLACEHOLDER,
        );
        std::fs::write(&path, header).map_err(|source| WriterError::Io {
            path: path.clone(),
            source,
        })?;

        self.path = Some(path);
        self.started_at = Some(now);
        Ok(self.path.as_deref().unwrap())
    }

    /// Append one timestamped segment to the active transcript.
    pub fn add_segment(&self, elapsed_seconds: f64, text: &str) -> Result<(), WriterError> {
        self.add_labeled_segment(elapsed_seconds, None, text)
    }

    /// Append one timestamped segment, optionally attributed to a speaker.
    ///
    /// Meetings capture the microphone and the system output as separate
    /// tracks, so the live path knows which side of the call each segment
    /// came from without any model. The individual voices inside a track are
    /// a separate, offline job — see [`crate::transcription::diarize`] — and
    /// reach this writer only through the importer, which passes the name it
    /// found.
    pub fn add_labeled_segment(
        &self,
        elapsed_seconds: f64,
        speaker: Option<&str>,
        text: &str,
    ) -> Result<(), WriterError> {
        let path = self.path.as_ref().ok_or(WriterError::NotStarted)?;
        let text = text.trim();
        if text.is_empty() {
            return Ok(());
        }
        let line = match speaker {
            Some(s) => format!("**[{}] {}:** {}\n\n", fmt_mmss(elapsed_seconds), s, text),
            None => format!("**[{}]** {}\n\n", fmt_mmss(elapsed_seconds), text),
        };
        let mut f = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|source| WriterError::Io {
                path: path.clone(),
                source,
            })?;
        f.write_all(line.as_bytes())
            .map_err(|source| WriterError::Io {
                path: path.clone(),
                source,
            })
    }

    /// Replace the duration placeholder in the header with the final value.
    pub fn finalize(&self, total_duration_seconds: f64) -> Result<(), WriterError> {
        let Some(path) = self.path.as_ref() else {
            return Ok(());
        };
        let content = std::fs::read_to_string(path).map_err(|source| WriterError::Io {
            path: path.clone(),
            source,
        })?;
        let replaced = content.replace(
            &format!("- **Duration:** {}", DURATION_PLACEHOLDER),
            &format!("- **Duration:** {}", fmt_hhmmss(total_duration_seconds)),
        );
        std::fs::write(path, replaced).map_err(|source| WriterError::Io {
            path: path.clone(),
            source,
        })
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

fn slugify(title: &str) -> String {
    let lower = title.trim().to_lowercase();
    let re = Regex::new(r"[^a-z0-9]+").expect("static regex");
    let cleaned = re.replace_all(&lower, "-");
    let trimmed = cleaned.trim_matches('-');
    let truncated: String = trimmed.chars().take(40).collect();
    if truncated.is_empty() {
        "meeting".into()
    } else {
        truncated
    }
}

fn fmt_mmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", s / 60, s % 60)
}

fn fmt_hhmmss(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    format!("{}h {:02}m {:02}s", s / 3600, (s % 3600) / 60, s % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn writes_segments_and_finalizes() {
        let dir = tempdir().unwrap();
        let mut w = TranscriptWriter::new(dir.path()).unwrap();
        let path = w.start_new(Some("Standup 13.04.")).unwrap().to_path_buf();
        w.add_segment(0.0, "Guten Morgen alle.").unwrap();
        w.add_segment(12.5, "Wir starten mit den Updates.").unwrap();
        w.add_labeled_segment(30.0, Some("Others"), "Klingt gut.")
            .unwrap();
        w.finalize(125.0).unwrap();

        let body = std::fs::read_to_string(&path).unwrap();
        assert!(body.contains("**[00:00]** Guten Morgen alle."));
        assert!(body.contains("**[00:12]** Wir starten mit den Updates."));
        assert!(body.contains("- **Duration:** 0h 02m 05s"));
        assert!(body.contains("**[00:30] Others:** Klingt gut."));
        assert!(!body.contains(DURATION_PLACEHOLDER));
    }

    #[test]
    fn slugify_collapses_punctuation() {
        assert_eq!(slugify("Standup 13.04."), "standup-13-04");
        assert_eq!(slugify("   !!!   "), "meeting");
    }
}
