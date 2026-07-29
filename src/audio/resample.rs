//! Downmix + sample-rate conversion shared by every capture path.
//!
//! Every source we capture (microphone via cpal, system audio via a Core
//! Audio process tap) arrives at the device's native rate with an
//! arbitrary channel count. The transcription engines all want 16 kHz mono
//! `f32`, so each source owns a [`Resampler`] that does the conversion
//! incrementally, block by block, keeping enough state to be seamless
//! across block boundaries.
//!
//! Decimating (e.g. 48 kHz → 16 kHz) without first removing everything
//! above the new Nyquist frequency folds that content back down as
//! aliasing noise. That noise lands right in the band the acoustic model
//! cares about, so we low-pass first with a 4th-order Butterworth and only
//! then interpolate.

/// Sample rate every transcription backend operates at.
pub const TARGET_RATE: u32 = 16_000;

/// Low-pass corner, a little under the 8 kHz Nyquist of [`TARGET_RATE`] so
/// the filter has room to roll off before it folds.
const CUTOFF_HZ: f32 = 7_200.0;

/// Q values for the two biquad sections that make up a 4th-order
/// Butterworth response.
const BUTTERWORTH_Q: [f32; 2] = [0.541_196_1, 1.306_562_9];

/// Direct-form-II transposed biquad.
#[derive(Clone, Copy, Default)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// RBJ cookbook low-pass section.
    fn lowpass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let w0 = std::f32::consts::TAU * cutoff / sample_rate;
        let (sin, cos) = w0.sin_cos();
        let alpha = sin / (2.0 * q);
        let a0 = 1.0 + alpha;
        let b = (1.0 - cos) / 2.0;
        Self {
            b0: b / a0,
            b1: (1.0 - cos) / a0,
            b2: b / a0,
            a1: (-2.0 * cos) / a0,
            a2: (1.0 - alpha) / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }
}

/// Streaming interleaved-any-rate → mono-16 kHz converter.
///
/// One instance per capture source. Not `Sync`; the owning stream callback
/// is the only thing that touches it.
pub struct Resampler {
    in_rate: u32,
    channels: usize,
    /// `None` when the source is already at or below [`TARGET_RATE`] and
    /// no anti-alias filtering is needed.
    lowpass: Option<[Biquad; 2]>,
    /// Ratio of input frames consumed per output sample.
    step: f64,
    /// Read position of the next output sample, relative to the start of
    /// the block currently being processed.
    pos: f64,
    /// Final filtered sample of the previous block. Interpolating the
    /// first output sample of a block needs the sample immediately before
    /// it, which lives in the block we already released.
    prev: Option<f32>,
    mono: Vec<f32>,
    buf: Vec<f32>,
}

impl Resampler {
    pub fn new(in_rate: u32, channels: usize) -> Self {
        let channels = channels.max(1);
        let in_rate = in_rate.max(1);
        // Only decimation aliases. Upsampling from a lower rate cannot
        // fold anything down, so skip the filter and its group delay.
        let lowpass = (in_rate > TARGET_RATE)
            .then(|| BUTTERWORTH_Q.map(|q| Biquad::lowpass(in_rate as f32, CUTOFF_HZ, q)));
        Self {
            in_rate,
            channels,
            lowpass,
            step: in_rate as f64 / TARGET_RATE as f64,
            pos: 0.0,
            prev: None,
            mono: Vec::new(),
            buf: Vec::new(),
        }
    }

    pub fn in_rate(&self) -> u32 {
        self.in_rate
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Convert one block of interleaved samples, appending the 16 kHz mono
    /// result to `out`.
    pub fn push_interleaved<F>(&mut self, data: &[F], out: &mut Vec<f32>)
    where
        F: Copy + Into<f32>,
    {
        self.mono.clear();
        self.mono.reserve(data.len() / self.channels + 1);
        for frame in data.chunks_exact(self.channels) {
            let sum: f32 = frame.iter().copied().map(Into::into).sum();
            self.mono.push(sum / self.channels as f32);
        }
        self.push_mono_internal(out);
    }

    /// Convert one block that is already mono.
    pub fn push_mono(&mut self, data: &[f32], out: &mut Vec<f32>) {
        self.mono.clear();
        self.mono.extend_from_slice(data);
        self.push_mono_internal(out);
    }

    /// Convert one block held as `channels` separate planar buffers. Core
    /// Audio hands tap audio over this way.
    pub fn push_planar(&mut self, planes: &[&[f32]], out: &mut Vec<f32>) {
        let Some(frames) = planes.iter().map(|p| p.len()).min() else {
            return;
        };
        self.mono.clear();
        self.mono.reserve(frames);
        let n = planes.len().max(1) as f32;
        for f in 0..frames {
            let sum: f32 = planes.iter().map(|p| p[f]).sum();
            self.mono.push(sum / n);
        }
        self.push_mono_internal(out);
    }

    fn push_mono_internal(&mut self, out: &mut Vec<f32>) {
        if self.mono.is_empty() {
            return;
        }

        if let Some(sections) = self.lowpass.as_mut() {
            for s in self.mono.iter_mut() {
                let mut y = *s;
                for section in sections.iter_mut() {
                    y = section.process(y);
                }
                *s = y;
            }
        }

        // Prepend the carried sample so every interpolation index is
        // non-negative, then walk the block at `step`.
        self.buf.clear();
        match self.prev {
            Some(p) => self.buf.push(p),
            // First block ever: start exactly on sample 0.
            None => self.pos = 0.0,
        }
        self.buf.extend_from_slice(&self.mono);

        let len = self.buf.len();
        while self.pos + 1.0 < len as f64 {
            let i = self.pos.floor() as usize;
            let frac = (self.pos - i as f64) as f32;
            let a = self.buf[i];
            let b = self.buf[i + 1];
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }

        // Carry the tail sample and rebase `pos` onto it so the fractional
        // phase survives into the next block without drifting.
        if let Some(&last) = self.buf.last() {
            self.prev = Some(last);
            self.pos = (self.pos - (len - 1) as f64).max(0.0);
        }
    }
}

/// Peak-preserving mix of two mono 16 kHz sources of possibly different
/// length. Shorter input is treated as silence past its end.
///
/// Averaging halves the level of whichever source is speaking alone, which
/// costs real headroom when only one side is talking — the common case in
/// a meeting. Summing and then normalising only if we actually clipped
/// keeps a lone speaker at full scale.
pub fn mix(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    let mut peak = 0.0f32;
    for i in 0..n {
        let s = a.get(i).copied().unwrap_or(0.0) + b.get(i).copied().unwrap_or(0.0);
        peak = peak.max(s.abs());
        out.push(s);
    }
    if peak > 1.0 {
        let g = 1.0 / peak;
        for s in out.iter_mut() {
            *s *= g;
        }
    }
    out
}

/// Smoothed RMS suitable for a UI level meter, in `[0, 1]`.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|s| s * s).sum();
    (sum_sq / samples.len() as f32 + 1e-12).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeding N seconds in must yield ~N seconds out, regardless of how
    /// the input is chopped into blocks.
    #[test]
    fn output_length_tracks_duration() {
        for &(rate, block) in &[(48_000u32, 512usize), (44_100, 441), (16_000, 160)] {
            let mut r = Resampler::new(rate, 1);
            let mut out = Vec::new();
            let total = rate as usize * 2; // two seconds
            let mut fed = 0;
            while fed < total {
                let n = block.min(total - fed);
                r.push_mono(&vec![0.0; n], &mut out);
                fed += n;
            }
            let expected = (TARGET_RATE * 2) as usize;
            let drift = out.len().abs_diff(expected);
            assert!(
                drift <= 4,
                "rate={rate} block={block}: got {} samples, expected ~{expected}",
                out.len()
            );
        }
    }

    /// Block boundaries must not change the result — chopping the same
    /// signal differently should produce the same output.
    #[test]
    fn blocking_is_transparent() {
        let signal: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();

        let mut one_shot = Vec::new();
        Resampler::new(48_000, 1).push_mono(&signal, &mut one_shot);

        let mut chunked = Vec::new();
        let mut r = Resampler::new(48_000, 1);
        for block in signal.chunks(333) {
            r.push_mono(block, &mut chunked);
        }

        assert_eq!(one_shot.len(), chunked.len());
        for (i, (a, b)) in one_shot.iter().zip(&chunked).enumerate() {
            assert!((a - b).abs() < 1e-5, "sample {i}: {a} vs {b}");
        }
    }

    #[test]
    fn interleaved_downmixes_to_mono() {
        let mut r = Resampler::new(16_000, 2);
        let mut out = Vec::new();
        // Hard-panned: left at +1, right at -1 cancels to silence.
        let data: Vec<f32> = (0..320)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        r.push_mono(&[], &mut out);
        r.push_interleaved(&data, &mut out);
        assert!(out.iter().all(|s| s.abs() < 1e-6), "channels should cancel");
    }

    #[test]
    fn mix_preserves_lone_speaker_level() {
        let a = vec![0.9f32; 64];
        let b = vec![0.0f32; 64];
        let m = mix(&a, &b);
        assert!(
            (m[0] - 0.9).abs() < 1e-6,
            "lone source was attenuated: {}",
            m[0]
        );
    }

    #[test]
    fn mix_normalises_only_on_clip() {
        let a = vec![0.8f32; 64];
        let b = vec![0.8f32; 64];
        let m = mix(&a, &b);
        assert!(m.iter().all(|s| *s <= 1.0 + 1e-6), "mix clipped");
        assert!(
            (m[0] - 1.0).abs() < 1e-6,
            "expected normalise to full scale"
        );
    }

    #[test]
    fn mix_handles_empty_and_ragged() {
        assert_eq!(mix(&[], &[1.0, 2.0]), vec![1.0, 2.0]);
        assert_eq!(mix(&[1.0, 2.0], &[]), vec![1.0, 2.0]);
        assert_eq!(mix(&[0.5, 0.5], &[0.5]).len(), 2);
    }
}
