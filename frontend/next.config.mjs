/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable PWA features
  experimental: {
    webpackBuildWorker: true,
  },
  eslint: {
    // No ESLint config/deps in this project, and Next 16 removed `next lint`.
    // Adding ESLint requires new devDependencies (and a pnpm-lock update), so
    // build-time lint stays off until that's set up deliberately.
    ignoreDuringBuilds: true,
  },
  typescript: {
    // Type errors now fail the build (tsc --noEmit is clean). This prevents
    // type regressions from shipping silently. Run `npx tsc --noEmit` locally.
    ignoreBuildErrors: false,
  },
  images: {
    unoptimized: true,
  },
}

export default nextConfig
