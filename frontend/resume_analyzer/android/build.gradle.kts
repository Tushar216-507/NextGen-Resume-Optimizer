// ✅ No Flutter plugin here, Flutter is included via settings.gradle.kts

plugins {
    id("com.android.application") version "8.3.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.22" apply false
    id("dev.flutter.flutter-gradle-plugin")
}

pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
        // Flutter SDK
        includeBuild("../flutter_tools/gradle")
    }
}




// Repositories are configured in settings.gradle.kts

