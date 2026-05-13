set shell := ["bash", "-cu"]

app_name      := "WhisprKing"
bundle_id     := "com.devmaxde.whisprking"
version       := "0.1.0"

bin           := "target/release/whisprking"
bin_debug     := "target/debug/whisprking"

bundle_dir    := "target/release/bundle"
app_bundle    := bundle_dir + "/" + app_name + ".app"
install_root  := "/Applications"
installed_app := install_root + "/" + app_name + ".app"

# Default: show available recipes
default:
    @just --list

# --- build -----------------------------------------------------------------

# Build release binary + ad-hoc codesign so Accessibility trust survives rebuilds
build:
    cargo build --release
    codesign --force --sign - {{bin}}
    @echo "signed: {{bin}}"
    @codesign -dv {{bin}} 2>&1 | sed 's/^/  /'

# Debug build + sign (slower runtime, faster compile)
build-debug:
    cargo build
    codesign --force --sign - {{bin_debug}}

# Release build with sherpa-onnx backend (NVIDIA NeMo Parakeet). First build ~10 min.
build-sherpa:
    cargo build --release --features sherpa
    codesign --force --sign - {{bin}}
    @echo "signed: {{bin}}"

# --- bundle ----------------------------------------------------------------

# Wrap an already-built release binary into a .app. Independent — fails if binary missing.
bundle: build
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -f "{{bin}}" ]; then
        echo "missing {{bin}} — run 'just build' first" >&2
        exit 1
    fi
    rm -rf "{{app_bundle}}"
    mkdir -p "{{app_bundle}}/Contents/MacOS"
    mkdir -p "{{app_bundle}}/Contents/Resources"
    cp "{{bin}}" "{{app_bundle}}/Contents/MacOS/{{app_name}}"
    cat > "{{app_bundle}}/Contents/Info.plist" <<PLIST
    <?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
    <plist version="1.0">
    <dict>
        <key>CFBundleName</key><string>{{app_name}}</string>
        <key>CFBundleDisplayName</key><string>{{app_name}}</string>
        <key>CFBundleIdentifier</key><string>{{bundle_id}}</string>
        <key>CFBundleVersion</key><string>{{version}}</string>
        <key>CFBundleShortVersionString</key><string>{{version}}</string>
        <key>CFBundleExecutable</key><string>{{app_name}}</string>
        <key>CFBundlePackageType</key><string>APPL</string>
        <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
        <key>LSMinimumSystemVersion</key><string>12.0</string>
        <key>LSUIElement</key><true/>
        <key>NSHighResolutionCapable</key><true/>
        <key>NSMicrophoneUsageDescription</key><string>{{app_name}} needs the microphone to transcribe speech.</string>
        <key>NSSpeechRecognitionUsageDescription</key><string>{{app_name}} processes speech locally for dictation.</string>
    </dict>
    </plist>
    PLIST
    codesign --force --deep --sign - "{{app_bundle}}"
    echo "bundled: {{app_bundle}}"

# --- install ---------------------------------------------------------------

# Copy an existing .app into /Applications. Independent — fails if bundle missing.
install: bundle
    #!/usr/bin/env bash
    set -euo pipefail
    if [ ! -d "{{app_bundle}}" ]; then
        echo "missing {{app_bundle}} — run 'just bundle' first" >&2
        exit 1
    fi
    rm -rf "{{installed_app}}"
    cp -R "{{app_bundle}}" "{{install_root}}/"
    echo "installed: {{installed_app}}"

# Remove the installed .app
uninstall:
    rm -rf {{installed_app}}
    @echo "removed: {{installed_app}}"

# --- combined --------------------------------------------------------------

# Full pipeline: build release binary, bundle into .app, install to /Applications
release: build bundle install
    @echo "release ready: {{installed_app}}"

# --- run / dev -------------------------------------------------------------

# Build, sign, run with default logging
run: build
    RUST_LOG="whisprking=info,info" ./{{bin}}

# Run with the sherpa-onnx backend enabled
run-sherpa: build-sherpa
    RUST_LOG="whisprking=info,info" ./{{bin}}

# Build, sign, run with verbose tracing + backtraces
trace: build
    RUST_LOG="whisprking=trace" RUST_BACKTRACE=full ./{{bin}}

# --- maintenance -----------------------------------------------------------

# Clear every macOS Accessibility entry. Next launch triggers fresh prompt.
reset-trust:
    tccutil reset Accessibility
    @echo "All Accessibility entries reset. Run 'just run' and accept the prompt."

# Full reset: clear trust + clean build + sign + run
fresh: reset-trust
    cargo clean
    just run

test:
    cargo test --lib

check:
    cargo check
    cargo clippy --all-targets -- -D warnings

clean:
    cargo clean
    rm -rf {{bundle_dir}}
