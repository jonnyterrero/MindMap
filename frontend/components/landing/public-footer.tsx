import Link from "next/link";
import { Brain } from "lucide-react";

export function PublicFooter() {
  const year = new Date().getFullYear();
  return (
    <footer className="border-t border-white/20 py-10">
      <div className="container mx-auto flex max-w-5xl flex-col gap-6 px-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Brain className="h-5 w-5 text-primary" aria-hidden="true" />
          <span className="font-semibold font-[family-name:var(--font-space-grotesk)]">
            MindMap
          </span>
        </div>

        <nav aria-label="Footer" className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground">
          <Link href="/login" className="hover:text-foreground">Sign in</Link>
          <Link href="/signup" className="hover:text-foreground">Create account</Link>
          <Link href="/privacy" className="hover:text-foreground">Privacy</Link>
          <Link href="/terms" className="hover:text-foreground">Terms</Link>
          <Link href="/support" className="hover:text-foreground">Support</Link>
        </nav>

        <p className="text-xs text-muted-foreground">
          &copy; {year} MindMap. Self-tracking, not medical advice.
        </p>
      </div>
    </footer>
  );
}
