import { expect, type Page } from "@playwright/test";

async function completeConsentIfNeeded(page: Page) {
  if (!page.url().includes("/consent")) return;

  for (const label of ["Terms of Service", "Privacy Policy", "Data Sharing"]) {
    await page.getByLabel(label).check();
  }

  await page.getByRole("button", { name: "Get Started" }).click();
  await page.waitForURL((url) => !url.pathname.includes("/consent"), { timeout: 30_000 });
}

async function completeOnboardingIfNeeded(page: Page) {
  if (!page.url().includes("/onboarding")) return;

  await page.getByRole("button", { name: "Continue" }).click();

  await page
    .getByText("I understand MindMap does not diagnose, treat, cure, or prevent disease")
    .click();
  await page.getByRole("button", { name: "Continue" }).click();

  await page.getByRole("button", { name: "Mood stability" }).click();
  await page.getByRole("button", { name: "Continue" }).click();

  await page.getByRole("button", { name: "Continue" }).click();

  await page.getByRole("button", { name: "Start Day 1 Check-In" }).click();
  await page.waitForURL((url) => !url.pathname.includes("/onboarding"), { timeout: 30_000 });
}

export async function ensureAppReady(page: Page) {
  await page.waitForURL(/\/(today|home|consent|onboarding)(\/)?$/, { timeout: 30_000 });

  await completeConsentIfNeeded(page);
  await completeOnboardingIfNeeded(page);

  await expect(page).toHaveURL(/\/(today|home)(\/)?$/);
  await expect(page.getByRole("link", { name: "MindMap" })).toBeVisible();
}
