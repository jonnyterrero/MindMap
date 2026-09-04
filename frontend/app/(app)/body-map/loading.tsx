import { CanvasSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Body map is an interactive body diagram beside a sensation panel.
export default function BodyMapLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <CanvasSkeleton className="h-96" />
      <Skeleton className="h-24 w-full rounded-xl" />
    </PageSkeleton>
  );
}
