//! macOS activation policy — what makes the main window a *real* window.
//!
//! A bundle with `LSUIElement` (or an app that calls
//! `setActivationPolicy:NSApplicationActivationPolicyAccessory`) is an
//! *agent*: menu-bar only. Its windows are not owned by a normal app, so
//! they get no Dock tile, are skipped by Cmd+Tab and Mission Control, and
//! third-party window managers cannot place or resize them — the window
//! just floats.
//!
//! WhisprKing wants both halves: a normal, managed window while it is on
//! screen, and a menu-bar-only background agent once the window is closed
//! and only the hotkey keeps running. So the policy is switched at runtime
//! instead of being pinned in `Info.plist`:
//!
//!   * window shown  → `Regular`   (Dock tile, Cmd+Tab, app menu, WM-managed)
//!   * window hidden → `Accessory` (no Dock tile, tray + hotkey stay alive)
//!
//! Keeping `Accessory` while hidden also protects dictation: the overlay
//! viewport must never make WhisprKing the active app, or the synthesized
//! Cmd+V would land in our own window instead of the one the user is
//! typing into.
//!
//! Both functions are no-ops off macOS and must be called on the main
//! thread (they are: every caller is inside the egui update loop).

/// Become a normal foreground app and bring the window forward.
pub fn set_regular() {
	imp::set_regular();
}

/// Drop back to a menu-bar-only background agent.
pub fn set_accessory() {
	imp::set_accessory();
}

#[cfg(target_os = "macos")]
mod imp {
	use std::ffi::c_void;
	use std::os::raw::c_char;
	use std::sync::atomic::{AtomicI8, Ordering};

	// NSApplicationActivationPolicy
	const REGULAR: isize = 0;
	const ACCESSORY: isize = 1;

	/// Last policy we set, so repeated shows/hides do not re-enter AppKit
	/// every frame. `-1` = unknown (nothing set yet).
	static CURRENT: AtomicI8 = AtomicI8::new(-1);

	// The process already links AppKit through winit and tray-icon; this is
	// here so the symbols are guaranteed present no matter what those crates
	// do in future versions.
	#[link(name = "AppKit", kind = "framework")]
	extern "C" {}

	extern "C" {
		fn objc_getClass(name: *const c_char) -> *mut c_void;
		fn sel_registerName(name: *const c_char) -> *mut c_void;
		fn objc_msgSend();
	}

	/// `objc_msgSend` is variadic in the headers but must be called through
	/// a pointer typed exactly like the method being sent, otherwise the
	/// arguments land in the wrong registers. Every selector below returns
	/// a pointer or a `BOOL`, so no `objc_msgSend_stret` case exists here.
	unsafe fn class(name: &[u8]) -> *mut c_void {
		objc_getClass(name.as_ptr() as *const c_char)
	}

	unsafe fn sel(name: &[u8]) -> *mut c_void {
		sel_registerName(name.as_ptr() as *const c_char)
	}

	unsafe fn shared_app() -> *mut c_void {
		let cls = class(b"NSApplication\0");
		if cls.is_null() {
			return std::ptr::null_mut();
		}
		let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
			std::mem::transmute(objc_msgSend as *const ());
		send(cls, sel(b"sharedApplication\0"))
	}

	fn set_policy(policy: isize) {
		if CURRENT.load(Ordering::Relaxed) == policy as i8 {
			return;
		}
		unsafe {
			let app = shared_app();
			if app.is_null() {
				log::warn!("activation policy: no NSApplication");
				return;
			}

			let send_policy: unsafe extern "C" fn(*mut c_void, *mut c_void, isize) -> i8 =
				std::mem::transmute(objc_msgSend as *const ());
			send_policy(app, sel(b"setActivationPolicy:\0"), policy);

			// Going Regular only *allows* a Dock tile and a menu bar; the app
			// still has to activate for them to show up and for the window to
			// come to the front.
			if policy == REGULAR {
				let send_activate: unsafe extern "C" fn(*mut c_void, *mut c_void, i8) =
					std::mem::transmute(objc_msgSend as *const ());
				send_activate(app, sel(b"activateIgnoringOtherApps:\0"), 1);
			}
		}
		CURRENT.store(policy as i8, Ordering::Relaxed);
		log::debug!(
			"activation policy → {}",
			if policy == REGULAR { "regular" } else { "accessory" }
		);
	}

	pub fn set_regular() {
		set_policy(REGULAR);
	}

	pub fn set_accessory() {
		set_policy(ACCESSORY);
	}
}

#[cfg(not(target_os = "macos"))]
mod imp {
	pub fn set_regular() {}
	pub fn set_accessory() {}
}
