/** @type {import('next').NextConfig} */
const nextConfig = {
  // Enable PWA features
  experimental: {
    webpackBuildWorker: true,
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
