import { test, expect } from "@playwright/test";
import { getE2ECredentials } from "./helpers/credentials";

test.describe("authenticated app", () => {
  test.beforeEach(() => {
    test.skip(
      !getE2ECredentials().configured,
      "Set E2E_TEST_EMAIL and E2E_TEST_PASSWORD in .env.local"
    );
  });

  test("home page loads with primary navigation", async ({ page }) => {
    await page.goto("/home");

    await expect(page).toHaveURL(/\/home$/);
    await expect(page.getByRole("link", { name: "MindMap" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Today" })).toBeVisible();
    await expect(page.getByText(/Today's check-in/)).toBeVisible();
  });

  test("today check-in page loads", async ({ page }) => {
    await page.goto("/today");

    await expect(page).toHaveURL(/\/today$/);
    await expect(
      page.getByText(/How are you doing today|Your check-in for today/)
    ).toBeVisible();
  });

  test("can navigate from home to today via nav", async ({ page }) => {
    await page.goto("/home");
    await page.getByRole("link", { name: "Today" }).click();

    await expect(page).toHaveURL(/\/today$/);
  });
});
