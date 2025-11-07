"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Download, Sparkles } from "lucide-react"

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[]
  readonly userChoice: Promise<{
    outcome: "accepted" | "dismissed"
    platform: string
  }>
  prompt(): Promise<void>
}

export function InstallButton() {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null)
  const [isInstalled, setIsInstalled] = useState(false)
  const [isInstalling, setIsInstalling] = useState(false)

  useEffect(() => {
    // Check if app is already installed
    if (window.matchMedia("(display-mode: standalone)").matches) {
      setIsInstalled(true)
    }

    // Listen for the beforeinstallprompt event
    const handleBeforeInstallPrompt = (e: Event) => {
      e.preventDefault()
      setDeferredPrompt(e as BeforeInstallPromptEvent)
    }

    // Listen for app installed event
    const handleAppInstalled = () => {
      setIsInstalled(true)
      setDeferredPrompt(null)
    }

    window.addEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
    window.addEventListener("appinstalled", handleAppInstalled)

    return () => {
      window.removeEventListener("beforeinstallprompt", handleBeforeInstallPrompt)
      window.removeEventListener("appinstalled", handleAppInstalled)
    }
  }, [])

  const handleInstallClick = async () => {
    if (!deferredPrompt) return

    setIsInstalling(true)

    try {
      await deferredPrompt.prompt()
      const { outcome } = await deferredPrompt.userChoice

      if (outcome === "accepted") {
        setIsInstalled(true)
      }

      setDeferredPrompt(null)
    } catch (error) {
      console.error("Error during installation:", error)
    } finally {
      setIsInstalling(false)
    }
  }

  // Don't show button if already installed or prompt not available
  if (isInstalled || (!deferredPrompt && !isInstalling)) {
    return null
  }

  return (
    <Button
      onClick={handleInstallClick}
      disabled={isInstalling}
      size="lg"
      className="glass relative overflow-hidden bg-gradient-to-r from-pink-500/40 to-purple-500/40 hover:from-pink-500/60 hover:to-purple-500/60 border-2 border-pink-400/50 text-white font-semibold shadow-2xl shadow-pink-500/25 backdrop-blur-md transition-all duration-500 animate-pulse hover:animate-none hover:scale-105 hover:shadow-pink-500/40"
    >
      <div className="absolute inset-0 bg-gradient-to-r from-pink-400/20 to-purple-400/20 animate-pulse" />
      {isInstalling ? (
        <>
          <div className="animate-spin rounded-full h-5 w-5 border-2 border-current border-t-transparent mr-3" />
          <span className="relative z-10">Installing...</span>
        </>
      ) : (
        <>
          <Sparkles className="h-5 w-5 mr-2 animate-pulse relative z-10" />
          <Download className="h-5 w-5 mr-2 relative z-10" />
          <span className="relative z-10">Install MindTrack</span>
        </>
      )}
      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 animate-shimmer" />
    </Button>
  )
}
