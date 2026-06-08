import { defineConfig, devices } from "@playwright/test";
import path from "path";
import { loadLocalEnv } from "./e2e/helpers/load-env";

loadLocalEnv();

const PORT = process.env.PLAYWRIGHT_PORT ?? "3000";
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? `http://localhost:${PORT}`;
const hasE2ECredentials = Boolean(
  process.env.E2E_TEST_EMAIL && process.env.E2E_TEST_PASSWORD
);
const authFile = path.join(__dirname, "e2e/.auth/user.json");

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },
  projects: [
    ...(hasE2ECredentials
      ? [
          {
            name: "setup",
            testMatch: /auth\.setup\.ts/,
          },
        ]
      : []),
    {
      name: "chromium",
      testIgnore: /auth\.setup\.ts|authenticated\.spec\.ts/,
      use: { ...devices["Desktop Chrome"] },
    },
    ...(hasE2ECredentials
      ? [
          {
            name: "chromium-authenticated",
            testMatch: /authenticated\.spec\.ts/,
            dependencies: ["setup"],
            use: {
              ...devices["Desktop Chrome"],
              storageState: authFile,
            },
          },
        ]
      : []),
  ],
  webServer: {
    command: "next dev",
    url: baseURL,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
