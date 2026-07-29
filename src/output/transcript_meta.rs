//! Sidecar JSON next to each transcript `.md` recording which derived
//! documents exist and when they were generated. Lives at `<stem>.meta.json`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Provenance of one derived document (cleanup, summary, …).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocMeta {
    /// Local timestamp the document was generated at.
    #[serde(default)]
    pub created_at: String,
    /// Model id that produced it, if known.
    #[serde(default)]
    pub model: String,
    /// Provider that produced it, if known.
    #[serde(default)]
    pub provider: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptMeta {
    /// Derived documents by preset key (`cleanup`, `summary`, …).
    #[serde(default)]
    pub docs: BTreeMap<String, DocMeta>,

    /// Legacy: set by builds that appended the cleanup into the transcript.
    /// Kept so old sidecars keep parsing; nothing writes it any more.
    #[serde(default)]
    pub cleanup_ran: bool,
    #[serde(default)]
    pub cleanup_at: Option<String>,
    #[serde(default)]
    pub cleanup_preset: Option<String>,
}

impl TranscriptMeta {
    /// Note that `preset` has just been generated.
    pub fn record_doc(&mut self, preset: &str, created_at: &str, model: &str, provider: &str) {
        if preset.is_empty() {
            return;
        }
        self.docs.insert(
            preset.to_string(),
            DocMeta {
                created_at: created_at.to_string(),
                model: model.to_string(),
                provider: provider.to_string(),
            },
        );
    }

    pub fn forget_doc(&mut self, preset: &str) {
        self.docs.remove(preset);
    }
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
        let mut m = TranscriptMeta::default();
        m.record_doc("cleanup", "2026-05-13 10:00", "gpt", "openrouter");
        m.save(&md).unwrap();

        let loaded = TranscriptMeta::load(&md);
        let doc = loaded.docs.get("cleanup").expect("recorded doc");
        assert_eq!(doc.created_at, "2026-05-13 10:00");
        assert_eq!(doc.provider, "openrouter");
        assert!(dir.path().join("2026-05-13_meeting.meta.json").is_file());
    }

    /// Sidecars written by builds that appended the cleanup into the
    /// transcript must keep parsing.
    #[test]
    fn legacy_sidecar_still_parses() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("old.md");
        std::fs::write(
            dir.path().join("old.meta.json"),
            r#"{"cleanup_ran": true, "cleanup_at": "2026-05-13T10:00:00", "cleanup_preset": "cleanup"}"#,
        )
        .unwrap();
        let m = TranscriptMeta::load(&md);
        assert!(m.cleanup_ran);
        assert!(m.docs.is_empty());
    }

    #[test]
    fn missing_returns_default() {
        let dir = tempdir().unwrap();
        let md = dir.path().join("nope.md");
        let m = TranscriptMeta::load(&md);
        assert!(m.docs.is_empty());
    }
}
