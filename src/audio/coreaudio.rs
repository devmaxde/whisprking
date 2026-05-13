//! Core Audio HAL bindings for the bits we need to build BlackHole
//! aggregate / multi-output devices.
//!
//! We intentionally hand-roll the FFI instead of pulling in `coreaudio-sys`
//! because we only need a handful of symbols and want to keep the dep tree
//! small. The aggregate-device dictionary string keys (`"uid"`, `"name"`,
//! `"master"`, …) are documented in `<CoreAudio/AudioHardware.h>` and have
//! been stable for many macOS releases.

#![cfg(target_os = "macos")]
#![allow(non_snake_case, non_upper_case_globals)]

use std::ffi::c_void;
use std::ptr;

use core_foundation::array::CFArray;
use core_foundation::base::{CFType, TCFType};
use core_foundation::dictionary::CFDictionary;
use core_foundation::number::CFNumber;
use core_foundation::string::{CFString, CFStringRef};

pub type OSStatus = i32;
pub type AudioObjectID = u32;
pub type AudioDeviceID = AudioObjectID;

pub const kAudioObjectSystemObject: AudioObjectID = 1;
pub const kAudioObjectUnknown: AudioObjectID = 0;

#[repr(C)]
struct AudioObjectPropertyAddress {
    mSelector: u32,
    mScope: u32,
    mElement: u32,
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

const fn fourcc(s: &[u8; 4]) -> u32 {
    ((s[0] as u32) << 24) | ((s[1] as u32) << 16) | ((s[2] as u32) << 8) | (s[3] as u32)
}

const kAudioHardwarePropertyDevices: u32 = fourcc(b"dev#");
const kAudioHardwarePropertyDefaultOutputDevice: u32 = fourcc(b"dOut");
const kAudioHardwarePropertyTranslateUIDToDevice: u32 = fourcc(b"uidd");
const kAudioDevicePropertyDeviceUID: u32 = fourcc(b"uid ");
const kAudioDevicePropertyDeviceNameCFString: u32 = fourcc(b"lnam");
const kAudioDevicePropertyStreamConfiguration: u32 = fourcc(b"slay");
const kAudioObjectPropertyScopeGlobal: u32 = fourcc(b"glob");
const kAudioObjectPropertyScopeInput: u32 = fourcc(b"inpt");
const kAudioObjectPropertyScopeOutput: u32 = fourcc(b"outp");
const kAudioObjectPropertyElementMain: u32 = 0;

#[link(name = "CoreAudio", kind = "framework")]
extern "C" {
    fn AudioObjectGetPropertyDataSize(
        inObjectID: AudioObjectID,
        inAddress: *const AudioObjectPropertyAddress,
        inQualifierDataSize: u32,
        inQualifierData: *const c_void,
        outDataSize: *mut u32,
    ) -> OSStatus;

    fn AudioObjectGetPropertyData(
        inObjectID: AudioObjectID,
        inAddress: *const AudioObjectPropertyAddress,
        inQualifierDataSize: u32,
        inQualifierData: *const c_void,
        ioDataSize: *mut u32,
        outData: *mut c_void,
    ) -> OSStatus;

    fn AudioObjectSetPropertyData(
        inObjectID: AudioObjectID,
        inAddress: *const AudioObjectPropertyAddress,
        inQualifierDataSize: u32,
        inQualifierData: *const c_void,
        inDataSize: u32,
        inData: *const c_void,
    ) -> OSStatus;

    fn AudioHardwareCreateAggregateDevice(
        inDescription: *const c_void, // CFDictionaryRef
        outDeviceID: *mut AudioObjectID,
    ) -> OSStatus;

    fn AudioHardwareDestroyAggregateDevice(inDeviceID: AudioObjectID) -> OSStatus;
}

#[derive(Debug, thiserror::Error)]
pub enum AudioErr {
    #[error("Core Audio call returned OSStatus {0}")]
    Status(OSStatus),
    #[error("device not found: {0}")]
    NotFound(String),
    #[error("BlackHole not installed")]
    BlackHoleMissing,
}

fn check(status: OSStatus) -> Result<(), AudioErr> {
    if status == 0 {
        Ok(())
    } else {
        Err(AudioErr::Status(status))
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: AudioObjectID,
    pub uid: String,
    pub name: String,
    pub input_channels: u32,
    pub output_channels: u32,
}

impl DeviceInfo {
    pub fn is_input(&self) -> bool {
        self.input_channels > 0
    }
    pub fn is_output(&self) -> bool {
        self.output_channels > 0
    }
}

/// Enumerate every audio device the HAL knows about.
pub fn list_devices() -> Result<Vec<DeviceInfo>, AudioErr> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };

    let mut size: u32 = 0;
    unsafe {
        check(AudioObjectGetPropertyDataSize(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            &mut size,
        ))?;
    }

    let count = (size as usize) / std::mem::size_of::<AudioObjectID>();
    let mut ids = vec![0u32; count];
    unsafe {
        check(AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            &mut size,
            ids.as_mut_ptr().cast(),
        ))?;
    }

    let mut out = Vec::with_capacity(count);
    for id in ids {
        if let Some(info) = device_info(id) {
            out.push(info);
        }
    }
    Ok(out)
}

fn device_info(id: AudioObjectID) -> Option<DeviceInfo> {
    let uid = get_cfstring(
        id,
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
    )?;
    let name = get_cfstring(
        id,
        kAudioDevicePropertyDeviceNameCFString,
        kAudioObjectPropertyScopeGlobal,
    )
    .unwrap_or_else(|| "<unnamed>".into());
    let input_channels = channel_count(id, kAudioObjectPropertyScopeInput).unwrap_or(0);
    let output_channels = channel_count(id, kAudioObjectPropertyScopeOutput).unwrap_or(0);
    Some(DeviceInfo {
        id,
        uid,
        name,
        input_channels,
        output_channels,
    })
}

fn get_cfstring(id: AudioObjectID, selector: u32, scope: u32) -> Option<String> {
    let addr = AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut cf_ref: CFStringRef = ptr::null();
    let mut size = std::mem::size_of::<CFStringRef>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            (&mut cf_ref as *mut CFStringRef).cast(),
        )
    };
    if status != 0 || cf_ref.is_null() {
        return None;
    }
    let s = unsafe { CFString::wrap_under_create_rule(cf_ref) };
    Some(s.to_string())
}

fn channel_count(id: AudioObjectID, scope: u32) -> Option<u32> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioDevicePropertyStreamConfiguration,
        mScope: scope,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut size: u32 = 0;
    unsafe {
        if AudioObjectGetPropertyDataSize(id, &addr, 0, ptr::null(), &mut size) != 0 {
            return None;
        }
    }
    if size == 0 {
        return Some(0);
    }
    let mut buf = vec![0u8; size as usize];
    unsafe {
        if AudioObjectGetPropertyData(
            id,
            &addr,
            0,
            ptr::null(),
            &mut size,
            buf.as_mut_ptr().cast(),
        ) != 0
        {
            return None;
        }
    }
    let list = unsafe { &*(buf.as_ptr() as *const AudioBufferList) };
    let n = list.mNumberBuffers as usize;
    let buffers = unsafe { std::slice::from_raw_parts(list.mBuffers.as_ptr(), n) };
    Some(buffers.iter().map(|b| b.mNumberChannels).sum())
}

pub fn find_blackhole() -> Option<DeviceInfo> {
    list_devices()
        .ok()?
        .into_iter()
        .find(|d| d.name.to_lowercase().contains("blackhole") && d.is_input())
}

pub fn find_by_uid(uid: &str) -> Option<DeviceInfo> {
    list_devices().ok()?.into_iter().find(|d| d.uid == uid)
}

pub fn default_output_uid() -> Option<String> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let mut id: AudioObjectID = 0;
    let mut size = std::mem::size_of::<AudioObjectID>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            &mut size,
            (&mut id as *mut AudioObjectID).cast(),
        )
    };
    if status != 0 || id == kAudioObjectUnknown {
        return None;
    }
    get_cfstring(
        id,
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
    )
}

pub fn set_default_output_by_uid(uid: &str) -> Result<(), AudioErr> {
    let id = translate_uid_to_device(uid)?;
    set_default_output(id)
}

fn translate_uid_to_device(uid: &str) -> Result<AudioObjectID, AudioErr> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyTranslateUIDToDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let cf_uid = CFString::new(uid);
    let mut cf_ref: CFStringRef = cf_uid.as_concrete_TypeRef();
    let mut id: AudioObjectID = 0;
    let mut size = std::mem::size_of::<AudioObjectID>() as u32;
    let qualifier_size = std::mem::size_of::<CFStringRef>() as u32;
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            qualifier_size,
            (&mut cf_ref as *mut CFStringRef).cast(),
            &mut size,
            (&mut id as *mut AudioObjectID).cast(),
        )
    };
    drop(cf_uid);
    if status != 0 || id == kAudioObjectUnknown {
        return Err(AudioErr::NotFound(uid.into()));
    }
    Ok(id)
}

fn set_default_output(id: AudioObjectID) -> Result<(), AudioErr> {
    let addr = AudioObjectPropertyAddress {
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    };
    let size = std::mem::size_of::<AudioObjectID>() as u32;
    unsafe {
        check(AudioObjectSetPropertyData(
            kAudioObjectSystemObject,
            &addr,
            0,
            ptr::null(),
            size,
            (&id as *const AudioObjectID).cast(),
        ))
    }
}

/// Build the description CFDictionary then call HAL to create the device.
/// `stacked=true` makes a Multi-Output Device (audio goes to every sub),
/// `stacked=false` makes a regular Aggregate Device (one input stream
/// concatenating every sub).
pub fn create_aggregate(
    name: &str,
    uid: &str,
    main_sub_uid: &str,
    sub_uids: &[&str],
    stacked: bool,
) -> Result<DeviceInfo, AudioErr> {
    // Sub-device array: [{ "uid": <subuid> }, …]
    let sub_dicts: Vec<CFDictionary<CFString, CFType>> = sub_uids
        .iter()
        .map(|sub| {
            let k = CFString::new("uid");
            let v = CFString::new(sub).as_CFType();
            CFDictionary::from_CFType_pairs(&[(k, v)])
        })
        .collect();
    let subs_array =
        CFArray::from_CFTypes(&sub_dicts.iter().map(|d| d.as_CFType()).collect::<Vec<_>>());

    let one = CFNumber::from(1i32);
    let zero = CFNumber::from(0i32);

    let pairs: Vec<(CFString, CFType)> = vec![
        (CFString::new("name"), CFString::new(name).as_CFType()),
        (CFString::new("uid"), CFString::new(uid).as_CFType()),
        (
            CFString::new("master"),
            CFString::new(main_sub_uid).as_CFType(),
        ),
        ("private".into(), zero.as_CFType()),
        (
            CFString::new("stacked"),
            if stacked {
                one.as_CFType()
            } else {
                zero.as_CFType()
            },
        ),
        (CFString::new("subdevices"), subs_array.as_CFType()),
    ];
    let desc = CFDictionary::from_CFType_pairs(&pairs);

    let mut new_id: AudioObjectID = 0;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(desc.as_concrete_TypeRef().cast(), &mut new_id)
    };
    check(status)?;
    drop(desc);

    device_info(new_id).ok_or(AudioErr::NotFound(uid.into()))
}

pub fn destroy_aggregate_by_uid(uid: &str) -> Result<(), AudioErr> {
    let id = translate_uid_to_device(uid)?;
    unsafe { check(AudioHardwareDestroyAggregateDevice(id)) }
}
