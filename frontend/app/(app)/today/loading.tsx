import { PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";
import { Skeleton } from "@/components/ui/skeleton";

// Today is a single tall guided check-in card under a date header.
export default function TodayLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton titleClassName="w-56" />
      <Skeleton className="h-[28rem] w-full rounded-xl" />
    </PageSkeleton>
  );
}
