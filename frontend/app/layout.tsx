import type React from "react"
import type { Metadata, Viewport } from "next"
import { Space_Grotesk, DM_Sans } from "next/font/google"
import "./globals.css"
import { Analytics } from "@vercel/analytics/react"
import { UpdatePrompt } from "@/components/update-prompt"
import { THEME_INIT_SCRIPT } from "@/lib/themes"

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-space-grotesk",
})

const dmSans = DM_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-dm-sans",
})

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // No maximumScale/userScalable clamp: pinning the scale to 1 blocks pinch-zoom,
  // which fails WCAG 1.4.4 and is hostile to low-vision users on a health app.
  viewportFit: "cover",
  themeColor: "#8955bd",
}

export const metadata: Metadata = {
  title: "MindMap - Mental Health Companion",
  description: "Your comprehensive mental health and wellness tracking companion with advanced analytics",
  manifest: "/manifest.json",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    // `data-app-theme` is set below, before first paint, from the theme cookie.
    // suppressHydrationWarning: the server cannot know the attribute the script
    // is about to write, and that mismatch on <html> alone is expected.
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT_SCRIPT }} />
      </head>
      <body className={`font-sans ${spaceGrotesk.variable} ${dmSans.variable} antialiased`}>
        {children}
        <Analytics />
        <UpdatePrompt />
      </body>
    </html>
  )
}
