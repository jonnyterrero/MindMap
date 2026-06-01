import type { CapacitorConfig } from "@capacitor/cli";

/**
 * MindMap — Capacitor (hosted / remote mode)
 * ------------------------------------------
 * MindMap is a server-rendered Next.js app (Server Actions, middleware auth,
 * Supabase SSR cookies), so it CANNOT be statically exported. Instead the
 * native iOS/Android shell loads the live production site in a native WebView
 * while still exposing the Capacitor bridge for native plugins (push, health,
 * haptics, etc.).
 *
 * `webDir` (mobile/www) is only a branded offline fallback — at runtime the app
 * loads `server.url`. See MOBILE.md for the full build + submission guide.
 */
const config: CapacitorConfig = {
  appId: "com.heartwire.mindmap",
  appName: "MindMap",
  webDir: "mobile/www",
  server: {
    url: "https://getmindmapplus.app",
    cleartext: false,
    // Full-page navigations the WebView is allowed to perform (auth redirects,
    // email-confirmation links). Network XHR/fetch (e.g. Supabase) is unaffected.
    allowNavigation: [
      "getmindmapplus.app",
      "www.getmindmapplus.app",
      "getmindmapplus.com",
      "*.supabase.co",
    ],
  },
  ios: {
    contentInset: "always",
  },
  android: {
    backgroundColor: "#0b0f1a",
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 1200,
      backgroundColor: "#0b0f1a",
      showSpinner: false,
    },
    PushNotifications: {
      presentationOptions: ["badge", "sound", "alert"],
    },
  },
};

export default config;
