import { CanvasSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Dashboard leads with the migraine risk card, then stacked 30-day charts.
export default function DashboardLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <Skeleton className="h-24 w-full rounded-xl" />
      <CanvasSkeleton />
      <CanvasSkeleton />
    </PageSkeleton>
  );
}
