import { CardStackSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";

// Segment-level fallback: shown instantly while any (app) route streams its
// server-rendered content, and used by routes that have no loading.tsx of
// their own. A generic header + card stack; routes whose shape differs
// meaningfully (charts, canvases, narrow containers) define their own.
export default function AppLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <CardStackSkeleton count={3} />
    </PageSkeleton>
  );
}
