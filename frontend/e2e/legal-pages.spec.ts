import { test, expect } from "@playwright/test";

const publicLegalPages = [
  { path: "/privacy", heading: "Privacy Policy" },
  { path: "/terms", heading: "Terms of Service" },
  { path: "/medical-disclaimer", heading: "Medical Disclaimer" },
  { path: "/support", heading: "Support" },
  { path: "/data-deletion", heading: "Data Deletion" },
] as const;

test.describe("public legal pages", () => {
  for (const { path, heading } of publicLegalPages) {
    test(`${path} loads without authentication`, async ({ page }) => {
      await page.goto(path);

      await expect(page).toHaveURL(new RegExp(`${path}$`));
      await expect(page.getByRole("heading", { name: heading, level: 1 })).toBeVisible();
    });
  }
});
