# Shiva iOS Packaging (via Briefcase / PyTo)

This directory contains instructions to embed Shiva's Python engine into a native Apple iOS App Bundle (`.ipa`).

## How it Works
We bundle Python as a native Apple Framework. The Python files and compiled libraries (`requirements.txt`) are placed inside the iOS `.app` resources folder. Xcode compiles the App using Swift/Objective-C which initializes the Python environment.

## Integration Steps

1. **Add hardware permissions to your Xcode project `Info.plist`:**
```xml
<key>NSCameraUsageDescription</key>
<string>Shiva needs camera access to observe your surrounding environment.</string>
<key>NSMicrophoneUsageDescription</key>
<string>Shiva needs microphone access to listen to acoustic alarms or sound events.</string>
<key>NSBluetoothAlwaysUsageDescription</key>
<string>Shiva needs Bluetooth access to identify local peripheral changes.</string>
```

2. **Initialize Python and Call Shiva in Swift:**
```swift
import Python

// Retrieve path to embedded Python resources bundle
guard let resourcePath = Bundle.main.resourcePath else { return }
let pythonHome = "\(resourcePath)/python"
setenv("PYTHONHOME", pythonHome, 1)

Py_Initialize()

// Call Shiva perception modules
let shivaDetector = PyImport_ImportModule("src.body.perception.detector")
// Process returns directly in Swift
```

3. **Alternative Frameworks:**
If you prefer a higher-level cross-platform approach, you can compile this project using **BeeWare Briefcase** (`briefcase package ios`) which automates Xcode target setup, CocoaPod bindings, and embeds Shiva's package structure automatically.
