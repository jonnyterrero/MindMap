# 🚀 Vercel Deployment Guide

This guide covers deploying the Sleep & Stress Tracker PWA to Vercel.

## ✅ Vercel-Ready Configuration

### Files Configured for Vercel

1. **`vercel.json`** - Vercel-specific headers and configuration
   - Service worker headers (Content-Type, Cache-Control)
   - Manifest.json headers
   - Icon caching headers
   - Security headers (X-Content-Type-Options, X-Frame-Options, etc.)

2. **`next.config.ts`** - Next.js configuration (works with Vercel)
   - Headers for service worker and manifest
   - Image optimization settings

3. **`public/sw.js`** - Service worker optimized for Vercel's CDN
   - Network-first strategy for HTML (leverages Vercel's edge caching)
   - Cache-first strategy for static assets
   - Proper error handling for offline scenarios

## 📋 Pre-Deployment Checklist

### 1. Environment Variables

Set these in your Vercel project settings:

**Required:**
- `DATABASE_URL` or `TURSO_CONNECTION_URL` - Database connection string
- `TURSO_AUTH_TOKEN` - Turso authentication token (if using Turso)

**Optional:**
- `NEXT_PUBLIC_APP_URL` - Your app's public URL (for API calls)

### 2. Generate PWA Icons

Before deploying, generate all required icons:

```bash
npm install sharp --save-dev
npm run generate-icons
```

Or manually create PNG icons in `public/`:
- `icon-72x72.png` through `icon-512x512.png`

### 3. Build Test

Test the build locally:

```bash
npm run build
npm run start
```

Verify:
- Build completes without errors
- Service worker registers
- Manifest is accessible
- All icons load

## 🚀 Deployment Steps

### Option 1: Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```

4. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

### Option 2: GitHub Integration

1. **Connect Repository**:
   - Go to [vercel.com](https://vercel.com)
   - Click "Add New Project"
   - Import your GitHub repository

2. **Configure Project**:
   - Framework Preset: **Next.js** (auto-detected)
   - Root Directory: `.` (or your project root)
   - Build Command: `npm run build` (default)
   - Output Directory: `.next` (default)

3. **Set Environment Variables**:
   - Go to Project Settings > Environment Variables
   - Add all required variables
   - Deploy

### Option 3: Vercel Dashboard

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your Git repository
3. Configure environment variables
4. Deploy

## 🔧 Vercel-Specific Optimizations

### Automatic Features

Vercel automatically provides:
- ✅ HTTPS (required for PWA)
- ✅ Edge Network (global CDN)
- ✅ Automatic deployments on git push
- ✅ Preview deployments for PRs
- ✅ Analytics and monitoring

### Service Worker on Vercel

The service worker is configured to:
- Work with Vercel's edge network
- Use network-first for HTML (benefits from Vercel's edge caching)
- Cache static assets efficiently
- Handle offline scenarios gracefully

### Headers Configuration

The `vercel.json` file ensures:
- Service worker has correct MIME type
- Manifest.json is properly cached
- Icons are cached for performance
- Security headers are set

## 🧪 Post-Deployment Testing

### 1. Verify PWA Files

Check these URLs on your deployed site:
- `https://your-domain.vercel.app/manifest.json` - Should return manifest
- `https://your-domain.vercel.app/sw.js` - Should return service worker
- `https://your-domain.vercel.app/icon-192x192.png` - Should return icon

### 2. Test Service Worker

1. Open DevTools > Application > Service Workers
2. Verify service worker is registered
3. Check for any errors

### 3. Test PWA Installation

- **Chrome/Edge**: Look for install button in address bar
- **Android**: Should show install prompt
- **iOS**: Manual install via Share > Add to Home Screen

### 4. Test Offline Mode

1. Open DevTools > Network
2. Enable "Offline" mode
3. Refresh page - should still work
4. Navigate to different pages - should work from cache

### 5. Lighthouse Audit

Run Lighthouse PWA audit:
1. Open DevTools > Lighthouse
2. Select "Progressive Web App"
3. Run audit
4. Should score 90+ for PWA

## 🔍 Troubleshooting

### Service Worker Not Registering

**Issue**: Service worker fails to register on Vercel

**Solutions**:
- Verify `sw.js` is in `public/` directory
- Check `vercel.json` headers are correct
- Ensure HTTPS is enabled (Vercel does this automatically)
- Check browser console for errors

### Icons Not Loading

**Issue**: Icons return 404 on Vercel

**Solutions**:
- Verify all icon files exist in `public/` directory
- Check file names are exact (case-sensitive)
- Ensure icons are committed to git
- Clear Vercel cache and redeploy

### Manifest Not Accessible

**Issue**: Manifest.json returns 404 or wrong content type

**Solutions**:
- Verify `manifest.json` is in `public/` directory
- Check `vercel.json` headers configuration
- Clear browser cache
- Redeploy if needed

### Build Failures

**Issue**: Build fails on Vercel

**Solutions**:
- Check build logs in Vercel dashboard
- Verify all dependencies are in `package.json`
- Ensure Node.js version is compatible (Vercel auto-detects)
- Check for TypeScript errors locally first

## 📊 Monitoring

### Vercel Analytics

Enable Vercel Analytics to monitor:
- Page views
- Performance metrics
- User behavior

### Error Tracking

Consider adding:
- Sentry for error tracking
- LogRocket for session replay
- Custom error logging

## 🔄 Continuous Deployment

### Automatic Deployments

Vercel automatically deploys:
- **Production**: Pushes to `main`/`master` branch
- **Preview**: Pull requests and other branches

### Custom Domains

1. Go to Project Settings > Domains
2. Add your custom domain
3. Configure DNS as instructed
4. SSL certificate is automatic

## 🎯 Performance Tips

1. **Image Optimization**: Use Next.js Image component
2. **Static Assets**: Leverage Vercel's edge caching
3. **API Routes**: Use Vercel's serverless functions efficiently
4. **Database**: Use connection pooling for database connections

## 📚 Additional Resources

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js on Vercel](https://vercel.com/docs/frameworks/nextjs)
- [Vercel Deployment Guide](https://vercel.com/docs/deployments/overview)
- [PWA on Vercel](https://vercel.com/docs/deployments/configuration#progressive-web-apps)

## ✅ Deployment Checklist

Before going live:

- [ ] All environment variables set in Vercel
- [ ] PWA icons generated and committed
- [ ] Build passes locally
- [ ] Service worker registers correctly
- [ ] Manifest.json accessible
- [ ] Offline mode tested
- [ ] Install prompt works
- [ ] Lighthouse PWA score > 90
- [ ] Custom domain configured (if applicable)
- [ ] Analytics enabled (optional)

---

**Ready to deploy!** 🚀

Your PWA is configured and optimized for Vercel. Just generate the icons and deploy!

