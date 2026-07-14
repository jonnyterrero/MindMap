# PWA Setup Guide

This guide explains how to complete the PWA setup for the Sleep & Stress Tracker app.

## ✅ Completed Setup

The following PWA features have been configured:

1. **Manifest File** (`public/manifest.json`) - App metadata and configuration
2. **Service Worker** (`public/sw.js`) - Offline functionality and caching
3. **PWA Components**:
   - `PWARegister.tsx` - Registers the service worker
   - `PWAInstallPrompt.tsx` - Shows install prompt to users
4. **Meta Tags** - Added to `layout.tsx` for PWA support
5. **Next.js Config** - Updated with PWA headers

## 🔧 Required: Generate PWA Icons

The app needs PNG icons in multiple sizes. You have two options:

### Option 1: Use the Icon Generator Script (Recommended)

1. Install sharp (if not already installed):
   ```bash
   npm install sharp --save-dev
   # or
   bun add -d sharp
   ```

2. Run the icon generator:
   ```bash
   node scripts/generate-icons.js
   ```

This will generate all required icon sizes from `public/icon.svg`.

### Option 2: Manual Icon Creation

Create PNG icons in the following sizes and place them in the `public/` directory:

- `icon-72x72.png`
- `icon-96x96.png`
- `icon-128x128.png`
- `icon-144x144.png`
- `icon-152x152.png`
- `icon-192x192.png`
- `icon-384x384.png`
- `icon-512x512.png`

You can use the provided `public/icon.svg` as a base and convert it using:
- Online tools (e.g., CloudConvert, Convertio)
- Image editing software (Photoshop, GIMP, etc.)
- Command line tools (ImageMagick, Inkscape)

## 🚀 Testing the PWA

### Local Development

1. Build the app:
   ```bash
   npm run build
   npm run start
   # or
   bun run build
   bun run start
   ```

2. Open in a browser that supports PWAs (Chrome, Edge, Safari)

3. Check the browser console for service worker registration messages

4. Open DevTools > Application tab:
   - Check "Manifest" section
   - Check "Service Workers" section
   - Verify icons are loading

### Testing PWA Features

1. **Install Prompt**: Should appear automatically on supported browsers
2. **Offline Mode**: 
   - Open DevTools > Network tab
   - Enable "Offline" mode
   - Refresh the page - it should still work
3. **App Installation**:
   - Look for install button in browser address bar
   - Or use the install prompt dialog

## 📱 Platform-Specific Notes

### iOS (Safari)
- Icons should be 152x152px minimum
- Add to home screen manually (Share > Add to Home Screen)
- The app will work in standalone mode

### Android (Chrome)
- Automatic install prompt will appear
- Icons should be 192x192px and 512x512px minimum
- App will appear in app drawer

### Desktop (Chrome/Edge)
- Install button in address bar
- App will appear in applications menu

## 🔍 Verification Checklist

Before deploying to production, verify:

- [ ] All icon sizes are generated and in `public/` directory
- [ ] Manifest.json is accessible at `/manifest.json`
- [ ] Service worker is registered (check browser console)
- [ ] App works offline (test with network disabled)
- [ ] Install prompt appears on supported browsers
- [ ] Icons display correctly when installed
- [ ] App name and theme color are correct
- [ ] Build completes without errors

## 🐛 Troubleshooting

### Service Worker Not Registering
- Check browser console for errors
- Verify `sw.js` is accessible at `/sw.js`
- Ensure you're using HTTPS (or localhost for development)

### Icons Not Showing
- Verify all icon files exist in `public/` directory
- Check file names match exactly (case-sensitive)
- Clear browser cache and reload

### Install Prompt Not Appearing
- Ensure all PWA requirements are met (manifest, service worker, HTTPS)
- Check if app is already installed
- Try in incognito/private mode

## 📚 Additional Resources

- [MDN: Progressive Web Apps](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps)
- [Web.dev: PWA Checklist](https://web.dev/pwa-checklist/)
- [Next.js PWA Documentation](https://nextjs.org/docs/app/building-your-application/configuring/progressive-web-apps)

