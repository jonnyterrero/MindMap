"use client";

import { useEffect } from "react";

/**
 * Last-resort boundary: catches errors thrown in the root layout itself, where
 * the app shell/stylesheet may not be available. Replaces the entire document,
 * so it renders its own <html>/<body> and uses inline styles only.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Global error boundary caught:", error);
  }, [error]);

  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          minHeight: "100vh",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily:
            "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
          background: "#0b0b0f",
          color: "#e5e7eb",
          padding: "1.5rem",
        }}
      >
        <div style={{ maxWidth: 420, textAlign: "center" }}>
          <h1 style={{ fontSize: "1.25rem", margin: "0 0 0.5rem" }}>
            Something went wrong
          </h1>
          <p style={{ fontSize: "0.875rem", color: "#9ca3af", margin: "0 0 1.25rem" }}>
            The app hit an unexpected error. Try reloading.
          </p>
          <button
            onClick={() => reset()}
            style={{
              cursor: "pointer",
              borderRadius: "0.5rem",
              border: "1px solid #374151",
              background: "#111827",
              color: "#e5e7eb",
              padding: "0.5rem 1rem",
              fontSize: "0.875rem",
            }}
          >
            Reload
          </button>
          {error.digest && (
            <p style={{ fontSize: "0.75rem", color: "#6b7280", marginTop: "1rem" }}>
              Reference: {error.digest}
            </p>
          )}
        </div>
      </body>
    </html>
  );
}
