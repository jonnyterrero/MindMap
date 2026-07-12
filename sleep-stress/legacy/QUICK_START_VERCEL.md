# ⚡ Quick Start: Deploy to Vercel

## 🚀 Fastest Way to Deploy

### 1. Generate Icons (Required)
```bash
npm install sharp --save-dev
npm run generate-icons
```

### 2. Set Environment Variables in Vercel
Go to your Vercel project → Settings → Environment Variables:
- `TURSO_CONNECTION_URL` (or `DATABASE_URL`)
- `TURSO_AUTH_TOKEN` (if using Turso)

### 3. Deploy

**Option A: Via GitHub (Recommended)**
1. Push code to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your repository
4. Vercel auto-detects Next.js
5. Add environment variables
6. Deploy!

**Option B: Via CLI**
```bash
npm i -g vercel
vercel login
vercel --prod
```

### 4. Verify PWA
After deployment, check:
- ✅ `https://your-app.vercel.app/manifest.json` - Should return manifest
- ✅ `https://your-app.vercel.app/sw.js` - Should return service worker
- ✅ Open in Chrome → DevTools → Application → Service Workers (should see registered)

## ✅ What's Already Configured

- ✅ `vercel.json` - Headers and configuration
- ✅ Service worker optimized for Vercel's CDN
- ✅ Next.js config compatible with Vercel
- ✅ Security headers
- ✅ PWA manifest and meta tags

## 🎯 That's It!

Your PWA is ready to deploy. Just generate the icons and push to Vercel!

For detailed instructions, see `VERCEL_DEPLOYMENT.md`

