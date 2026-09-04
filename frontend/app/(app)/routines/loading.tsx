import { ListRowsSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Routines: an add form above the list of saved routines.
export default function RoutinesLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <Skeleton className="h-24 w-full rounded-xl" />
      <ListRowsSkeleton count={4} />
    </PageSkeleton>
  );
}
