"use client"

import type React from "react"

import { Toaster as Sonner } from "sonner"

type ToasterProps = React.ComponentProps<typeof Sonner>

const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      className="toaster group"
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-white group-[.toaster]:text-[var(--color-text)] group-[.toaster]:border-[var(--color-border)] group-[.toaster]:shadow-lg",
          description: "group-[.toast]:text-[var(--color-text-secondary)]",
          actionButton: "group-[.toast]:bg-[var(--color-primary)] group-[.toast]:text-white",
          cancelButton: "group-[.toast]:bg-white/60 group-[.toast]:text-[var(--color-text)]",
        },
      }}
      {...props}
    />
  )
}

export { Toaster }
