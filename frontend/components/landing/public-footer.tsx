import Link from "next/link";
import { Brain } from "lucide-react";

const LEGAL_LINKS = [
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/terms", label: "Terms of Service" },
  { href: "/medical-disclaimer", label: "Medical Disclaimer" },
  { href: "/ai-disclosure", label: "AI / ML Disclosure" },
  { href: "/data-deletion", label: "Data Deletion" },
  { href: "/support", label: "Support" },
  { href: "/crisis-resources", label: "Crisis Resources" },
];

export function PublicFooter() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-white/20 py-10">
      <div className="container mx-auto flex max-w-5xl flex-col gap-6 px-4">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-xs space-y-2">
            <div className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
              <span className="font-semibold font-[family-name:var(--font-space-grotesk)]">
                MindMap
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Private self-tracking for mood, sleep, migraines, and daily
              patterns. Not medical advice.
            </p>
          </div>

          <nav
            aria-label="Legal and support"
            className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm text-muted-foreground sm:grid-cols-2"
          >
            {LEGAL_LINKS.map((item) => (
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

        <div className="flex flex-col gap-2 border-t border-white/20 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-muted-foreground">
            &copy; {year} MindMap. Self-tracking and wellness reflection, not
            medical advice.
          </p>
          <div className="flex gap-4 text-xs text-muted-foreground">
            <Link href="/login" className="hover:text-foreground">
              Sign in
            </Link>
            <Link href="/signup" className="hover:text-foreground">
              Create account
            </Link>
          </div>
        </div>
      </div>
    </footer>
  );
}
