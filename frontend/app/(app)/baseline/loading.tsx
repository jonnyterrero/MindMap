import { PageHeaderSkeleton, PageSkeleton, StatGridSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Baseline is the one (app) route that narrows its container: it renders
// inside `mx-auto max-w-lg space-y-5`, so the shared skeleton's full-width
// spacing would visibly jump on load. Matches the eight stat tiles the
// unlocked report shows; the locked state is a single card, and either way
// this settles into the right column width.
export default function BaselineLoading() {
  return (
    <PageSkeleton className="mx-auto max-w-lg space-y-5">
      <PageHeaderSkeleton />
      <StatGridSkeleton count={8} />
      <Skeleton className="h-32 w-full rounded-xl" />
    </PageSkeleton>
  );
}
