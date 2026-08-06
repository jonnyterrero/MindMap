import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Compass, Home } from "lucide-react";

export const metadata = {
  title: "Page not found · MindMap",
};

// Root 404. Deliberately mirrors the RouteError card so a missing page and a
// failed page feel like the same product rather than two different apps.
//
// Renders outside the (app) group, so there is no sidebar to navigate with --
// hence the explicit links out. Signed-out visitors rarely reach this: the
// middleware sends unknown paths to /login before Next can resolve a 404.
export default function NotFound() {
  return (
    <div className="mx-auto max-w-lg px-4 py-16">
      <Card>
        <CardHeader className="text-center">
          <div className="mb-2 flex justify-center">
            <div className="rounded-full bg-muted p-3">
              <Compass className="h-7 w-7 text-muted-foreground" />
            </div>
          </div>
          <CardTitle>Page not found</CardTitle>
          <CardDescription>
            This page doesn&apos;t exist, or it may have moved. Nothing you&apos;ve
            logged has been affected.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex justify-center gap-2">
          <Button asChild variant="outline">
            <Link href="/today">Today&apos;s check-in</Link>
          </Button>
          <Button asChild>
            <Link href="/home">
              <Home /> Home
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
