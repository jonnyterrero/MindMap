import type React from "react"
import type { Metadata } from "next"
import { Inter, JetBrains_Mono } from "next/font/google"
import "./globals.css"
import { Navigation } from "@/components/layout/navigation"
import { HealthDataProvider } from "@/contexts/HealthDataContext"
import { Toaster } from "@/components/ui/sonner"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
})

export const metadata: Metadata = {
  title: "Health Tracker - AI-Powered Wellness",
  description: "Track your sleep, mood, and health with AI-powered insights",
    generator: 'v0.app'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased">
        <HealthDataProvider>
          <Navigation />
          <div className="pb-20 md:pb-0 md:pt-20">{children}</div>
          <Toaster />
        </HealthDataProvider>
      </body>
    </html>
  )
}
