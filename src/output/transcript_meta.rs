//! Sidecar JSON next to each transcript `.md` recording whether the
//! cleanup post-process has been run. Lives at `<stem>.meta.json`.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptMeta {
    #[serde(default)]
    pub cleanup_ran: bool,
    /// ISO-8601 local timestamp the cleanup completed at.
    #[serde(default)]
    pub cleanup_at: Option<String>,
    /// Preset key used (e.g. `cleanup`, `summary`).
    #[serde(default)]
    pub cleanup_preset: Option<String>,
}

impl TranscriptMeta {
    pub fn path_for(transcript: &Path) -> PathBuf {
        let stem = transcript
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_default();
        let mut name = stem;
        name.push(".meta.json");
        transcript.with_file_name(name)
    }

    pub fn load(transcript: &Path) -> Self {
        let p = Self::path_for(transcript);
        std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self, transcript: &Path) -> std::io::Result<()> {
        let p = Self::path_for(transcript);
        let body = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(p, body)
    }

    pub fn delete(transcript: &Path) {
        let _ = std::fs::remove_file(Self::path_for(transcript));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn round_trip() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("2026-05-13_meeting.md");
        std::fs::write(&md, "body").unwrap();
        let m = TranscriptMeta {
            cleanup_ran: true,
            cleanup_at: Some("2026-05-13T10:00:00".into()),
            cleanup_preset: Some("cleanup".into()),
        };
        m.save(&md).unwrap();
        let loaded = TranscriptMeta::load(&md);
        assert!(loaded.cleanup_ran);
        assert_eq!(loaded.cleanup_preset.as_deref(), Some("cleanup"));
        let meta_path = dir.path().join("2026-05-13_meeting.meta.json");
        assert!(meta_path.is_file());
    }

    #[test]
    fn missing_returns_default() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("nope.md");
        let m = TranscriptMeta::load(&md);
        assert!(!m.cleanup_ran);
    }
}
