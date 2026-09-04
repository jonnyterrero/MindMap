import { CardStackSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Journal is a composer above the list of past entries.
export default function JournalLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <Skeleton className="h-40 w-full rounded-xl" />
      <CardStackSkeleton count={3} />
    </PageSkeleton>
  );
}
