import { ListRowsSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Reports: the generate control above previously generated reports.
export default function ReportsLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <Skeleton className="h-20 w-full rounded-xl" />
      <ListRowsSkeleton count={3} />
    </PageSkeleton>
  );
}
