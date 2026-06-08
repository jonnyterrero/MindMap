import { test as setup, expect } from "@playwright/test";
import path from "path";
import { getE2ECredentials } from "./helpers/credentials";
import { ensureAppReady } from "./helpers/ensure-app-ready";

const authFile = path.join(__dirname, ".auth/user.json");

setup("authenticate test user", async ({ page }) => {
  const { email, password, configured } = getE2ECredentials();
  setup.skip(!configured, "Set E2E_TEST_EMAIL and E2E_TEST_PASSWORD in .env.local");

  await page.goto("/login");
  await page.getByLabel("Email").fill(email!);
  await page.getByLabel("Password").fill(password!);
  await page.getByRole("button", { name: "Sign In" }).click();

  await ensureAppReady(page);
  await expect(page.getByRole("link", { name: "Today" })).toBeVisible();

  await page.context().storageState({ path: authFile });
});
