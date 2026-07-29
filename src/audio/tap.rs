//! System-audio capture via Core Audio process taps (macOS 14.4+).
//!
//! A process tap hands us a read-only copy of what the system is playing.
//! Compared to the BlackHole approach this replaces, it has three
//! properties that matter a lot here:
//!
//! - **No virtual driver.** Nothing to install, nothing to configure in
//!   Audio MIDI Setup.
//! - **The user's devices are never touched.** We do not change the
//!   default output, we do not build a multi-output device, and we never
//!   ask the user to point their meeting app at a device we own. Whatever
//!   Zoom was using before stays exactly as it was.
//! - **The tap's aggregate is private** (`kAudioAggregateDeviceIsPrivateKey`),
//!   so it is visible only to this process. It cannot show up in another
//!   app's device picker and be mistaken for a microphone — which is
//!   precisely the failure that made callers hear their own audio played
//!   back at them instead of the user's voice.
//!
//! `CATapDescription` is Objective-C only; there is no C entry point for
//! building one. The ObjC surface we need is six selectors, so we hand-roll
//! it against the runtime rather than take an `objc2` dependency — matching
//! how `coreaudio.rs` already hand-rolls its HAL bindings. `NSString` is
//! toll-free bridged to `CFString`, so the one string we read back comes
//! out through `core-foundation` instead of a second Foundation binding.

#![cfg(target_os = "macos")]
#![allow(non_snake_case, non_upper_case_globals)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::{Arc, Mutex};

use core_foundation::array::CFArray;
use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};

use super::resample::Resampler;

pub type OSStatus = i32;
pub type AudioObjectID = u32;
type AudioDeviceIOProcID = *mut c_void;

const kAudioObjectUnknown: AudioObjectID = 0;

const fn fourcc(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | (s[3] as u32)
}

const kAudioTapPropertyFormat: u32 = fourcc(b"tfmt");
const kAudioObjectPropertyScopeGlobal: u32 = fourcc(b"glob");
const kAudioObjectPropertyElementMain: u32 = 0;

/// `kAudioFormatFlagIsFloat`
const kAudioFormatFlagIsFloat: u32 = 1 << 0;
/// `kAudioFormatFlagIsNonInterleaved`
const kAudioFormatFlagIsNonInterleaved: u32 = 1 << 5;

#[repr(C)]
struct AudioObjectPropertyAddress {
    mSelector: u32,
    mScope: u32,
    mElement: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AudioStreamBasicDescription {
    mSampleRate: f64,
    mFormatID: u32,
    mFormatFlags: u32,
    mBytesPerPacket: u32,
    mFramesPerPacket: u32,
    mBytesPerFrame: u32,
    mChannelsPerFrame: u32,
    mBitsPerChannel: u32,
    mReserved: u32,
}

#[repr(C)]
struct AudioBuffer {
    mNumberChannels: u32,
    mDataByteSize: u32,
    mData: *mut c_void,
}

#[repr(C)]
struct AudioBufferList {
    mNumberBuffers: u32,
    mBuffers: [AudioBuffer; 1],
}

#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyData(
        inObjectID: AudioObjectID,
        inAddress: *const AudioObjectPropertyAddress,
        inQualifierDataSize: u32,
        inQualifierData: *const c_void,
        ioDataSize: *mut u32,
        outData: *mut c_void,
    ) -> OSStatus;

    fn AudioHardwareCreateAggregateDevice(
        inDescription: *const c_void,
        outDeviceID: *mut AudioObjectID,
    ) -> OSStatus;

    fn AudioHardwareDestroyAggregateDevice(inDeviceID: AudioObjectID) -> OSStatus;

    fn AudioDeviceCreateIOProcID(
        inDevice: AudioObjectID,
        inProc: unsafe extern "C" fn(
            AudioObjectID,
            *const c_void,
            *const AudioBufferList,
            *const c_void,
            *mut AudioBufferList,
            *const c_void,
            *mut c_void,
        ) -> OSStatus,
        inClientData: *mut c_void,
        outIOProcID: *mut AudioDeviceIOProcID,
    ) -> OSStatus;

    fn AudioDeviceDestroyIOProcID(
        inDevice: AudioObjectID,
        inIOProcID: AudioDeviceIOProcID,
    ) -> OSStatus;

    fn AudioDeviceStart(inDevice: AudioObjectID, inProcID: AudioDeviceIOProcID) -> OSStatus;
    fn AudioDeviceStop(inDevice: AudioObjectID, inProcID: AudioDeviceIOProcID) -> OSStatus;
}

// The process-tap calls only exist on macOS 14.4+. Declaring them as
// ordinary externs would make the binary fail to *load* on an older
// system — dyld binds these eagerly — so both are resolved through
// `dlsym` and their absence is what `is_supported()` reports.
type CreateProcessTapFn = unsafe extern "C" fn(*mut c_void, *mut AudioObjectID) -> OSStatus;
type DestroyProcessTapFn = unsafe extern "C" fn(AudioObjectID) -> OSStatus;

extern "C" {
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

/// `RTLD_DEFAULT` is `(void *)-2` on macOS, **not** NULL as it is on Linux.
/// Passing NULL here searches nothing and every lookup silently fails.
const RTLD_DEFAULT: *mut c_void = -2isize as *mut c_void;

fn create_process_tap_fn() -> Option<CreateProcessTapFn> {
    unsafe {
        let sym = dlsym(RTLD_DEFAULT, c"AudioHardwareCreateProcessTap".as_ptr());
        (!sym.is_null()).then(|| std::mem::transmute::<*mut c_void, CreateProcessTapFn>(sym))
    }
}

fn destroy_process_tap_fn() -> Option<DestroyProcessTapFn> {
    unsafe {
        let sym = dlsym(RTLD_DEFAULT, c"AudioHardwareDestroyProcessTap".as_ptr());
        (!sym.is_null()).then(|| std::mem::transmute::<*mut c_void, DestroyProcessTapFn>(sym))
    }
}

fn destroy_process_tap(tap_id: AudioObjectID) {
    if let Some(f) = destroy_process_tap_fn() {
        let status = unsafe { f(tap_id) };
        if status != 0 {
            log::warn!("destroying audio tap {tap_id} returned OSStatus {status}");
        }
    }
}

// ---------------------------------------------------------------------
// Minimal Objective-C runtime access for CATapDescription
// ---------------------------------------------------------------------

type Id = *mut c_void;
type Sel = *mut c_void;
type Class = *mut c_void;

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> Class;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
}

/// `objc_msgSend` is variadic in the headers but must be called through a
/// pointer cast to the callee's exact signature — the arm64 ABI passes
/// variadic and non-variadic arguments differently, so calling it as
/// declared would corrupt the frame.
unsafe fn msg_send_id(receiver: Id, sel: Sel) -> Id {
    let f: unsafe extern "C" fn(Id, Sel) -> Id = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel)
}

unsafe fn msg_send_id_id(receiver: Id, sel: Sel, arg: Id) -> Id {
    let f: unsafe extern "C" fn(Id, Sel, Id) -> Id = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel, arg)
}

unsafe fn msg_send_void_id(receiver: Id, sel: Sel, arg: Id) {
    let f: unsafe extern "C" fn(Id, Sel, Id) = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel, arg)
}

unsafe fn msg_send_void_bool(receiver: Id, sel: Sel, arg: bool) {
    let f: unsafe extern "C" fn(Id, Sel, bool) = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel, arg)
}

unsafe fn msg_send_void_isize(receiver: Id, sel: Sel, arg: isize) {
    let f: unsafe extern "C" fn(Id, Sel, isize) = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel, arg)
}

unsafe fn msg_send_void(receiver: Id, sel: Sel) {
    let f: unsafe extern "C" fn(Id, Sel) = std::mem::transmute(objc_msgSend as *const ());
    f(receiver, sel)
}

fn sel(name: &CStr) -> Sel {
    unsafe { sel_registerName(name.as_ptr()) }
}

fn class(name: &CStr) -> Option<Class> {
    let c = unsafe { objc_getClass(name.as_ptr()) };
    (!c.is_null()).then_some(c)
}

/// `CATapUnmuted` — the tapped audio keeps playing to the user's speakers.
/// Muting here would silence the meeting for the user.
const CATapUnmuted: isize = 0;

/// Owned `CATapDescription`, released on drop.
struct TapDescription(Id);

impl TapDescription {
    /// Stereo mixdown of everything the system is playing. The exclusion
    /// list is empty, which for `…ButExcludeProcesses:` means "exclude
    /// nothing" — i.e. tap all output.
    fn global(name: &str) -> Result<Self, TapError> {
        let cls = class(c"CATapDescription").ok_or(TapError::Unsupported)?;
        let ns_array = class(c"NSArray").ok_or(TapError::Unsupported)?;

        unsafe {
            let empty = msg_send_id(ns_array, sel(c"array"));
            let alloc = msg_send_id(cls, sel(c"alloc"));
            if alloc.is_null() {
                return Err(TapError::DescriptionInit);
            }
            let desc = msg_send_id_id(
                alloc,
                sel(c"initStereoGlobalTapButExcludeProcesses:"),
                empty,
            );
            if desc.is_null() {
                return Err(TapError::DescriptionInit);
            }

            let ns_name = CFString::new(name);
            // CFStringRef is toll-free bridged to NSString*.
            msg_send_void_id(desc, sel(c"setName:"), ns_name.as_concrete_TypeRef() as Id);
            // Private: the tap and its aggregate stay invisible to every
            // other process on the system.
            msg_send_void_bool(desc, sel(c"setPrivate:"), true);
            msg_send_void_isize(desc, sel(c"setMuteBehavior:"), CATapUnmuted);

            Ok(Self(desc))
        }
    }

    /// UID string the aggregate description refers to the tap by.
    fn uuid_string(&self) -> Result<String, TapError> {
        unsafe {
            let uuid = msg_send_id(self.0, sel(c"UUID"));
            if uuid.is_null() {
                return Err(TapError::DescriptionInit);
            }
            let s = msg_send_id(uuid, sel(c"UUIDString"));
            if s.is_null() {
                return Err(TapError::DescriptionInit);
            }
            // NSString* → CFStringRef is a free bridge; the object is
            // autoreleased, so borrow it rather than taking ownership.
            let cf = CFString::wrap_under_get_rule(s as CFStringRef);
            Ok(cf.to_string())
        }
    }

    fn as_id(&self) -> Id {
        self.0
    }
}

impl Drop for TapDescription {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { msg_send_void(self.0, sel(c"release")) };
        }
    }
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum TapError {
    #[error(
        "system audio capture needs macOS 14.4 or newer — this Mac is running an older version"
    )]
    Unsupported,
    #[error("could not build the audio tap description")]
    DescriptionInit,
    #[error(
        "macOS denied system audio recording. Grant WhisprKing access under \
         System Settings → Privacy & Security → Screen & System Audio Recording, \
         then restart the app. (OSStatus {0})"
    )]
    PermissionDenied(OSStatus),
    #[error("creating the audio tap failed (OSStatus {0})")]
    CreateTap(OSStatus),
    #[error("creating the private tap device failed (OSStatus {0})")]
    CreateAggregate(OSStatus),
    #[error("reading the tap audio format failed (OSStatus {0})")]
    Format(OSStatus),
    #[error("starting system audio capture failed (OSStatus {0})")]
    Start(OSStatus),
}

/// Statuses the HAL returns when the audio-capture TCC permission has not
/// been granted. `!hog` is `kAudioDevicePermissionsError` and `nope` is
/// `kAudioHardwareIllegalOperationError`; both show up in practice
/// depending on where the check trips.
const kAudioDevicePermissionsError: OSStatus = fourcc(b"!hog") as OSStatus;
const kAudioHardwareIllegalOperationError: OSStatus = fourcc(b"nope") as OSStatus;

/// Is system-audio capture available on this machine at all?
///
/// Checks for the 14.4+ API rather than parsing an OS version string, so a
/// future rename or backport is handled correctly.
pub fn is_supported() -> bool {
    create_process_tap_fn().is_some() && class(c"CATapDescription").is_some()
}

/// State owned exclusively by the IOProc. Behind its own lock so the
/// callback can hold it for the whole invocation without blocking the
/// consumer thread draining `buffer`.
struct TapState {
    resampler: Resampler,
    /// Samples converted but not yet handed over, because `buffer` was
    /// momentarily locked by the consumer. Flushed on the next callback.
    ///
    /// The alternative — dropping them — costs words in the transcript,
    /// and blocking on a real-time audio thread risks a glitch. Carrying
    /// them forward does neither.
    carry: Vec<f32>,
}

/// Shared sink the real-time IOProc writes converted samples into.
struct TapShared {
    state: Mutex<TapState>,
    buffer: Mutex<Vec<f32>>,
    level: Mutex<f32>,
}

/// A live system-audio tap. Capture stops and every HAL object we created
/// is destroyed when this is dropped.
pub struct SystemAudioTap {
    tap_id: AudioObjectID,
    aggregate_id: AudioObjectID,
    proc_id: AudioDeviceIOProcID,
    shared: Arc<TapShared>,
    /// The `Arc` handed to the IOProc as `clientData`, reclaimed on drop
    /// only after the IOProc is guaranteed to have stopped.
    client_ptr: *const TapShared,
    format: AudioStreamBasicDescription,
}

// The HAL calls the IOProc on its own real-time thread; everything shared
// with it lives behind a Mutex inside `TapShared`.
unsafe impl Send for SystemAudioTap {}

impl SystemAudioTap {
    /// Create and start a global system-audio tap.
    pub fn start() -> Result<Self, TapError> {
        let create_tap = create_process_tap_fn().ok_or(TapError::Unsupported)?;

        let desc = TapDescription::global("WhisprKing System Audio")?;
        let tap_uid = desc.uuid_string()?;

        let mut tap_id: AudioObjectID = kAudioObjectUnknown;
        let status = unsafe { create_tap(desc.as_id(), &mut tap_id) };
        if status != 0 || tap_id == kAudioObjectUnknown {
            return Err(
                if status == kAudioDevicePermissionsError
                    || status == kAudioHardwareIllegalOperationError
                {
                    TapError::PermissionDenied(status)
                } else {
                    TapError::CreateTap(status)
                },
            );
        }

        let guard = TapGuard(tap_id);

        let format = tap_format(tap_id)?;
        log::info!(
            "system audio tap: {} Hz, {} ch, flags 0x{:x}",
            format.mSampleRate,
            format.mChannelsPerFrame,
            format.mFormatFlags,
        );

        let aggregate_id = create_tap_aggregate(&tap_uid)?;
        let agg_guard = AggregateGuard(aggregate_id);

        let shared = Arc::new(TapShared {
            state: Mutex::new(TapState {
                resampler: Resampler::new(
                    format.mSampleRate as u32,
                    format.mChannelsPerFrame as usize,
                ),
                carry: Vec::new(),
            }),
            buffer: Mutex::new(Vec::new()),
            level: Mutex::new(0.0),
        });

        // Hand the IOProc a raw +1 reference; reclaimed in Drop.
        let client_ptr = Arc::into_raw(Arc::clone(&shared));

        let mut proc_id: AudioDeviceIOProcID = ptr::null_mut();
        let status = unsafe {
            AudioDeviceCreateIOProcID(
                aggregate_id,
                tap_ioproc,
                client_ptr as *mut c_void,
                &mut proc_id,
            )
        };
        if status != 0 {
            unsafe { drop(Arc::from_raw(client_ptr)) };
            return Err(TapError::Start(status));
        }

        let status = unsafe { AudioDeviceStart(aggregate_id, proc_id) };
        if status != 0 {
            unsafe {
                AudioDeviceDestroyIOProcID(aggregate_id, proc_id);
                drop(Arc::from_raw(client_ptr));
            }
            return Err(TapError::Start(status));
        }

        // Everything is live — hand ownership to the returned struct.
        std::mem::forget(guard);
        std::mem::forget(agg_guard);

        Ok(Self {
            tap_id,
            aggregate_id,
            proc_id,
            shared,
            client_ptr,
            format,
        })
    }

    /// Native sample rate of the tapped stream, before conversion.
    pub fn source_rate(&self) -> u32 {
        self.format.mSampleRate as u32
    }

    /// Take everything captured since the last call, as 16 kHz mono.
    pub fn drain(&self) -> Vec<f32> {
        self.sink().drain()
    }

    /// Smoothed capture level in `[0, 1]` for the UI meter.
    pub fn level(&self) -> f32 {
        self.sink().level()
    }

    /// A `Send + Clone` handle onto the tap's buffer, for the dispatcher
    /// thread. Dropping every sink does not stop capture — only dropping
    /// the [`SystemAudioTap`] does.
    pub fn sink(&self) -> TapSink {
        TapSink(Arc::clone(&self.shared))
    }
}

/// Thread-safe view onto a tap's captured audio.
#[derive(Clone)]
pub struct TapSink(Arc<TapShared>);

impl TapSink {
    pub fn drain(&self) -> Vec<f32> {
        let mut guard = self.0.buffer.lock().expect("tap buffer");
        std::mem::take(&mut *guard)
    }

    pub fn level(&self) -> f32 {
        *self.0.level.lock().expect("tap level")
    }
}

impl Drop for SystemAudioTap {
    fn drop(&mut self) {
        // Order matters. Every HAL object that could still drive the
        // IOProc has to be gone before the `clientData` Arc is reclaimed,
        // or the callback dereferences freed memory.
        unsafe {
            AudioDeviceStop(self.aggregate_id, self.proc_id);
            AudioDeviceDestroyIOProcID(self.aggregate_id, self.proc_id);
            AudioHardwareDestroyAggregateDevice(self.aggregate_id);
        }
        destroy_process_tap(self.tap_id);
        unsafe { drop(Arc::from_raw(self.client_ptr)) };
        log::info!("system audio tap torn down");
    }
}

/// Destroys a tap if we bail out before it is owned by a [`SystemAudioTap`].
struct TapGuard(AudioObjectID);
impl Drop for TapGuard {
    fn drop(&mut self) {
        destroy_process_tap(self.0);
    }
}

struct AggregateGuard(AudioObjectID);
impl Drop for AggregateGuard {
    fn drop(&mut self) {
        unsafe { AudioHardwareDestroyAggregateDevice(self.0) };
    }
}

fn tap_format(tap_id: AudioObjectID) -> Result<AudioStreamBasicDescription, TapError> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut asbd = AudioStreamBasicDescription::default();
    let mut size = std::mem::size_of::<AudioStreamBasicDescription>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            tap_id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            (&mut asbd as *mut AudioStreamBasicDescription).cast(),
        )
    };
    if status != 0 || asbd.mSampleRate <= 0.0 {
        return Err(TapError::Format(status));
    }
    Ok(asbd)
}

/// Private aggregate device that owns the tap.
///
/// `private = 1` is the important flag: it keeps this device out of every
/// other process's device list, so nothing can select it as a microphone.
fn create_tap_aggregate(tap_uid: &str) -> Result<AudioObjectID, TapError> {
    // The aggregate needs a real device to clock off; a tap on its own has
    // no timebase and the IOProc may never fire. The current default
    // output is the natural choice since it is what the tap is capturing.
    //
    // Listing it as a sub-device does *not* take it away from the user:
    // the aggregate is private, and the tap is a read-only copy of the
    // stream that device is already playing.
    let output_uid = super::coreaudio::default_output_uid().unwrap_or_default();
    if output_uid.is_empty() {
        log::warn!("system audio tap: no default output device to clock against");
    }

    let one = CFNumber::from(1i32);
    let zero = CFNumber::from(0i32);

    let sub_tap = CFDictionary::from_CFType_pairs(&[
        (CFString::new("uid"), CFString::new(tap_uid).as_CFType()),
        // Drift compensation: the tap and the clock device are not
        // guaranteed to share a timebase over a long meeting.
        (CFString::new("drift"), one.as_CFType()),
    ]);
    let taps = CFArray::from_CFTypes(&[sub_tap.as_CFType()]);

    let sub_device = CFDictionary::from_CFType_pairs(&[(
        CFString::new("uid"),
        CFString::new(&output_uid).as_CFType(),
    )]);
    let subdevices = CFArray::from_CFTypes(&[sub_device.as_CFType()]);

    let pairs: Vec<(CFString, CFType)> = vec![
        (
            CFString::new("name"),
            CFString::new("WhisprKing System Audio").as_CFType(),
        ),
        (
            CFString::new("uid"),
            CFString::new("com.devmaxde.whisprking.tap").as_CFType(),
        ),
        (
            CFString::new("master"),
            CFString::new(&output_uid).as_CFType(),
        ),
        (CFString::new("private"), one.as_CFType()),
        (CFString::new("stacked"), zero.as_CFType()),
        (CFString::new("tapautostart"), one.as_CFType()),
        (CFString::new("subdevices"), subdevices.as_CFType()),
        (CFString::new("taps"), taps.as_CFType()),
    ];
    let desc = CFDictionary::from_CFType_pairs(&pairs);

    let mut id: AudioObjectID = kAudioObjectUnknown;
    let status =
        unsafe { AudioHardwareCreateAggregateDevice(desc.as_concrete_TypeRef().cast(), &mut id) };
    if status != 0 || id == kAudioObjectUnknown {
        return Err(TapError::CreateAggregate(status));
    }
    Ok(id)
}

/// Real-time callback. Keep it allocation-light and never block on
/// anything the UI thread holds for long.
unsafe extern "C" fn tap_ioproc(
    _device: AudioObjectID,
    _now: *const c_void,
    input: *const AudioBufferList,
    _input_time: *const c_void,
    _output: *mut AudioBufferList,
    _output_time: *const c_void,
    client: *mut c_void,
) -> OSStatus {
    if input.is_null() || client.is_null() {
        return 0;
    }
    let shared = &*(client as *const TapShared);
    let list = &*input;
    let n = list.mNumberBuffers as usize;
    if n == 0 {
        return 0;
    }
    let buffers = std::slice::from_raw_parts(list.mBuffers.as_ptr(), n);

    // Uncontended in practice — this callback is the only writer.
    let Ok(mut state) = shared.state.try_lock() else {
        return 0;
    };
    let state = &mut *state;
    let resampler = &mut state.resampler;

    let mut converted = Vec::new();

    if n > 1 {
        // Non-interleaved: one mono buffer per channel.
        let planes: Vec<&[f32]> = buffers
            .iter()
            .filter(|b| !b.mData.is_null())
            .map(|b| {
                std::slice::from_raw_parts(
                    b.mData as *const f32,
                    b.mDataByteSize as usize / std::mem::size_of::<f32>(),
                )
            })
            .collect();
        if planes.is_empty() {
            return 0;
        }
        resampler.push_planar(&planes, &mut converted);
    } else {
        let b = &buffers[0];
        if b.mData.is_null() {
            return 0;
        }
        let samples = std::slice::from_raw_parts(
            b.mData as *const f32,
            b.mDataByteSize as usize / std::mem::size_of::<f32>(),
        );
        if b.mNumberChannels <= 1 {
            resampler.push_mono(samples, &mut converted);
        } else {
            resampler.push_interleaved(samples, &mut converted);
        }
    }

    if converted.is_empty() && state.carry.is_empty() {
        return 0;
    }

    if !converted.is_empty() {
        let rms = super::resample::rms(&converted);
        if let Ok(mut level) = shared.level.try_lock() {
            *level = 0.75 * *level + 0.25 * (rms * 6.0).min(1.0);
        }
    }

    match shared.buffer.try_lock() {
        Ok(mut buf) => {
            // Anything held back by an earlier contended callback goes
            // first, so ordering is preserved.
            if !state.carry.is_empty() {
                buf.append(&mut state.carry);
            }
            buf.extend_from_slice(&converted);
        }
        Err(_) => state.carry.extend_from_slice(&converted),
    }
    0
}

/// Sanity note for maintainers: the tap always hands us float samples
/// (`kAudioFormatFlagIsFloat`). If a future macOS changes that, the
/// `*const f32` casts above become wrong — this asserts the assumption in
/// the log rather than silently producing noise.
pub fn warn_if_unexpected_format(tap: &SystemAudioTap) {
    let flags = tap.format.mFormatFlags;
    if flags & kAudioFormatFlagIsFloat == 0 {
        log::warn!(
            "system audio tap reported a non-float format (flags 0x{flags:x}); \
             captured audio may be garbage"
        );
    }
    if flags & kAudioFormatFlagIsNonInterleaved != 0 && tap.format.mChannelsPerFrame > 1 {
        log::debug!(
            "system audio tap is non-interleaved, {} planes",
            tap.format.mChannelsPerFrame
        );
    }
}
