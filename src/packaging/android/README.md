# Shiva Android Packaging (via Chaquopy)

This directory contains configuration guidelines to embed Shiva's Python engine into a native Android Studio application.

## How it Works
We use **Chaquopy**, a Python SDK for Android. It embeds a complete Python interpreter inside your native Android APK and translates dynamic calls between Kotlin/Java and Python.

## Integration Steps

1. **Add Chaquopy to your project-level build.gradle:**
```gradle
plugins {
    id 'com.chaquo.python' version '15.0.1' apply false
}
```

2. **Add permissions to AndroidManifest.xml:**
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.BLUETOOTH" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
```

3. **Initialize Python in your main Android Activity (Kotlin/Java):**
```kotlin
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform

if (!Python.isStarted()) {
    Python.start(AndroidPlatform(context))
}
```

4. **Invoke Shiva from your Android app code:**
```kotlin
val py = Python.getInstance()
val shivaModule = py.getModule("src.body.perception.detector")
val currentOs = shivaModule.callAttr("get_current_os").toString() // Returns "android"
```
