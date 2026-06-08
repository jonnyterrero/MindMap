import { test, expect } from "@playwright/test";

test.describe("public auth routes", () => {
  test("login page renders sign-in form", async ({ page }) => {
    await page.goto("/login");

    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByText("Welcome Back")).toBeVisible();
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByLabel("Password")).toBeVisible();
    await expect(page.getByRole("button", { name: "Sign In" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Forgot password?" })).toHaveAttribute(
      "href",
      "/forgot-password"
    );
  });

  test("signup page renders create-account form", async ({ page }) => {
    await page.goto("/signup");

    await expect(page).toHaveURL(/\/signup$/);
    await expect(page.locator('[data-slot="card-title"]')).toHaveText("Create Account");
    await expect(page.getByLabel("Email")).toBeVisible();
    await expect(page.getByLabel("Password", { exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "Create Account" })).toBeVisible();
  });

  test("unauthenticated users are redirected to login from protected routes", async ({
    page,
  }) => {
    await page.goto("/home");

    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByText("Welcome Back")).toBeVisible();
  });

  test("root path redirects unauthenticated users to login", async ({ page }) => {
    await page.goto("/");

    await expect(page).toHaveURL(/\/login$/);
  });
});
