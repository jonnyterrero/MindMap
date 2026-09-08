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

  test("journal page loads", async ({ page }) => {
    await page.goto("/journal");

    await expect(page).toHaveURL(/\/journal$/);
    await expect(page.getByRole("heading", { name: "Journal", level: 1 })).toBeVisible();
  });

  test("insights page loads", async ({ page }) => {
    await page.goto("/insights");

    await expect(page).toHaveURL(/\/insights$/);
    await expect(page.getByRole("heading", { name: "Insights", level: 1 })).toBeVisible();
    await expect(page.getByText(/not medical advice/i)).toBeVisible();
  });

  test("companion page loads", async ({ page }) => {
    await page.goto("/companion");

    await expect(page).toHaveURL(/\/companion$/);
    await expect(page.getByRole("heading", { name: "Companion", level: 1 })).toBeVisible();
  });

  test("settings shows data privacy deletion control", async ({ page }) => {
    await page.goto("/settings");

    await expect(page).toHaveURL(/\/settings$/);
    await expect(page.getByRole("heading", { name: "Settings", level: 1 })).toBeVisible();
    await expect(page.getByText("Your data & privacy")).toBeVisible();
    await expect(page.getByRole("button", { name: "Delete account" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Export" })).toBeVisible();
  });
});
