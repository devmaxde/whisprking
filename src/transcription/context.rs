//! Decode-context carryover shared by every multi-segment transcription
//! path (live meetings and imported recordings alike).
//!
//! Long audio is transcribed one [`SpeechSpan`](crate::audio::SpeechSpan) at
//! a time. Priming each segment with the tail of the previous one
//! (whisper's `initial_prompt`) keeps a chunk that continues a sentence
//! coherent instead of guessing at a mid-thought start. The one hazard is a
//! degenerate decode — a repetition loop — which must never be carried
//! forward, or it poisons every following segment.

/// Longest context carried into the next segment. Whisper only honours
/// ~224 tokens of `initial_prompt`, so keep the tail and let it truncate.
pub const PRIOR_MAX_CHARS: usize = 800;

/// Turn a configured meeting language into a per-call override for
/// [`DecodeContext::language`](crate::transcription::engine::DecodeContext).
///
/// Empty means "follow whatever the engine was built with" (the dictation
/// language), which is the default and preserves the behaviour from before
/// meetings had their own language. Anything else — including `"auto"` —
/// is an explicit choice and overrides the engine.
pub fn language_override(configured: &str) -> Option<&str> {
    (!configured.is_empty()).then_some(configured)
}

/// Replace `prior` with the tail of the segment just decoded, priming the
/// next segment on the same track with what led into it. A degenerate decode
/// (a repetition loop) is dropped rather than carried — priming the next
/// chunk with garbage would only spread the hallucination.
pub fn carry_context(prior: &mut String, text: &str) {
    prior.clear();
    if looks_degenerate(text) {
        return;
    }
    if text.len() <= PRIOR_MAX_CHARS {
        prior.push_str(text);
        return;
    }
    // Keep the last PRIOR_MAX_CHARS, cutting on a char boundary.
    let cut = text.len() - PRIOR_MAX_CHARS;
    let start = text
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| i >= cut)
        .unwrap_or(0);
    prior.push_str(&text[start..]);
}

/// Cheap repetition-loop detector: many words but very few distinct ones is
/// the signature of a whisper hallucination loop ("the the the …").
pub fn looks_degenerate(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 6 {
        return false;
    }
    let unique: std::collections::HashSet<&str> = words.iter().copied().collect();
    // 20% or fewer distinct words ⇒ treat as a loop.
    unique.len() * 5 <= words.len()
}

#[cfg(test)]
mod tests {
    use super::{carry_context, language_override, looks_degenerate, PRIOR_MAX_CHARS};

    #[test]
    fn language_override_distinguishes_unset_from_auto() {
        // Unset: leave the engine's own language alone.
        assert_eq!(language_override(""), None);
        // "auto" is a deliberate choice and must reach the engine, so a
        // meeting can auto-detect while dictation stays pinned.
        assert_eq!(language_override("auto"), Some("auto"));
        assert_eq!(language_override("de"), Some("de"));
    }

    #[test]
    fn degenerate_detects_repetition_loops() {
        assert!(looks_degenerate("the the the the the the the the"));
        assert!(looks_degenerate(
            "yeah yeah yeah yeah yeah yeah yeah yeah yeah okay"
        ));
    }

    #[test]
    fn degenerate_passes_normal_speech() {
        assert!(!looks_degenerate(
            "so the main thing we need to decide today is the release date"
        ));
        // Short utterances are never treated as loops.
        assert!(!looks_degenerate("okay sounds good"));
        assert!(!looks_degenerate(""));
    }

    #[test]
    fn carry_keeps_clean_tail_and_drops_garbage() {
        let mut prior = String::from("stale");
        carry_context(&mut prior, "let's move on to the budget");
        assert_eq!(prior, "let's move on to the budget");

        // A loop wipes the context instead of poisoning the next decode.
        carry_context(&mut prior, "and and and and and and and");
        assert!(prior.is_empty());
    }

    #[test]
    fn carry_caps_length_on_a_char_boundary() {
        // Multi-byte chars near the cut must not split a codepoint.
        let text = "ä".repeat(1000); // 2 bytes each ⇒ 2000 bytes
        let mut prior = String::new();
        carry_context(&mut prior, &text);
        assert!(prior.len() <= PRIOR_MAX_CHARS + 1);
        assert!(prior.chars().all(|c| c == 'ä'));
    }
}
