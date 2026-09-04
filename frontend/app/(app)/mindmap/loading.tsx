import { CanvasSkeleton, PageHeaderSkeleton, PageSkeleton } from "@/components/page-skeletons";

// Mindmap renders a graph canvas rather than a card stack.
export default function MindmapLoading() {
  return (
    <PageSkeleton>
      <PageHeaderSkeleton />
      <CanvasSkeleton className="h-96" />
    </PageSkeleton>
  );
}
