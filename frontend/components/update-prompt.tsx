"use client"

import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { RefreshCw } from "lucide-react"

export function UpdatePrompt() {
  const [showPrompt, setShowPrompt] = useState(false)
  const [registration, setRegistration] = useState<ServiceWorkerRegistration | null>(null)

  useEffect(() => {
    const isPreview =
      window.location.hostname.includes("vusercontent.net") || window.location.hostname.includes("vercel.app")
    const isLocalhost = window.location.hostname === "localhost"
    const isProduction = window.location.protocol === "https:" && !isPreview

    if (typeof window !== "undefined" && "serviceWorker" in navigator && (isLocalhost || isProduction)) {
      // Register service worker
      navigator.serviceWorker
        .register("/sw.js")
        .then((reg) => {
          setRegistration(reg)

          // Check for updates every 60 seconds
          setInterval(() => {
            reg.update()
          }, 60000)

          // Listen for updates
          reg.addEventListener("updatefound", () => {
            const newWorker = reg.installing
            if (newWorker) {
              newWorker.addEventListener("statechange", () => {
                if (newWorker.state === "installed" && navigator.serviceWorker.controller) {
                  // New service worker available
                  setShowPrompt(true)
                }
              })
            }
          })
        })
        .catch((error) => {
          console.log("[v0] Service Worker not available in this environment")
        })

      // Listen for controller change (new SW activated)
      navigator.serviceWorker.addEventListener("controllerchange", () => {
        window.location.reload()
      })
    } else {
      console.log("[v0] Service Worker disabled in preview environment")
    }
  }, [])

  const handleUpdate = () => {
    if (registration && registration.waiting) {
      // Tell the waiting service worker to skip waiting
      registration.waiting.postMessage({ type: "SKIP_WAITING" })
    }
  }

  if (!showPrompt) return null

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 flex items-center justify-between gap-4 rounded-xl border border-white/20 bg-white/10 p-4 shadow-lg backdrop-blur-xl md:left-auto md:right-4 md:w-96">
      <div className="flex-1">
        <p className="text-sm font-medium text-foreground">New version available!</p>
        <p className="text-xs text-muted-foreground">Update now to get the latest features</p>
      </div>
      <Button
        onClick={handleUpdate}
        size="sm"
        className="glass-button flex items-center gap-2 bg-gradient-to-r from-pink-400/80 to-purple-400/80 hover:from-pink-500/90 hover:to-purple-500/90"
      >
        <RefreshCw className="h-4 w-4" />
        Update
      </Button>
    </div>
  )
}
