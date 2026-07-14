import Link from "next/link";
import type React from "react";

const NAV = [
  { href: "/privacy", label: "Privacy" },
  { href: "/terms", label: "Terms" },
  { href: "/medical-disclaimer", label: "Disclaimer" },
  { href: "/ai-disclosure", label: "AI / ML" },
  { href: "/data-deletion", label: "Data Deletion" },
  { href: "/support", label: "Support" },
  { href: "/crisis-resources", label: "Crisis" },
];

export default function LegalLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border/60">
        <div className="container mx-auto flex max-w-3xl flex-wrap items-center justify-between gap-4 px-4 py-4">
          <Link href="/" className="font-semibold tracking-tight">
            MindMap
          </Link>
          <nav className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-muted-foreground">
            {NAV.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>
      <main className="container mx-auto max-w-3xl px-4 py-10">
        <article className="prose prose-neutral dark:prose-invert max-w-none [&_h1]:mb-2 [&_h2]:mt-8 [&_h2]:mb-3">
          {children}
        </article>
      </main>
      <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
        © {new Date().getFullYear()} MindMap · Heartwire
      </footer>
    </div>
  );
}
