// Script to generate PWA icons from SVG
// Run with: node scripts/generate-icons.js
// Requires: sharp (npm install sharp --save-dev)

const fs = require('fs');
const path = require('path');

// Icon sizes needed for PWA
const iconSizes = [72, 96, 128, 144, 152, 192, 384, 512];

async function generateIcons() {
  try {
    // Check if sharp is available
    let sharp;
    try {
      sharp = require('sharp');
    } catch (e) {
      console.log('Sharp not found. Installing...');
      console.log('Please run: npm install sharp --save-dev');
      console.log('Or: bun add -d sharp');
      return;
    }

    const svgPath = path.join(__dirname, '../public/icon.svg');
    const publicDir = path.join(__dirname, '../public');

    if (!fs.existsSync(svgPath)) {
      console.error('icon.svg not found in public directory');
      return;
    }

    console.log('Generating PWA icons...');

    for (const size of iconSizes) {
      const outputPath = path.join(publicDir, `icon-${size}x${size}.png`);
      
      await sharp(svgPath)
        .resize(size, size)
        .png()
        .toFile(outputPath);
      
      console.log(`✓ Generated icon-${size}x${size}.png`);
    }

    console.log('\n✅ All icons generated successfully!');
  } catch (error) {
    console.error('Error generating icons:', error);
    console.log('\nNote: You can also manually create PNG icons from icon.svg');
    console.log('Required sizes:', iconSizes.join(', '));
  }
}

generateIcons();

