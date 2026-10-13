const path = require('path');
const { pathToFileURL } = require('url');
const { chromium } = require('C:/Users/30491/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const root = path.resolve(__dirname, '../..');
(async () => {
  const browser = await chromium.launch({
    executablePath: 'C:/Users/30491/AppData/Local/ms-playwright/chromium_headless_shell-1228/chrome-headless-shell-win64/chrome-headless-shell.exe',
    headless: true,
  });
  try {
    const page = await browser.newPage({ viewport: { width: 2400, height: 1760 } });
    for (const lang of ['zh', 'en']) {
      const source = path.join(root, `paper/figures/v5/supplement/figure_s1_current_${lang}.svg`);
      await page.goto(pathToFileURL(source).href);
      await page.locator('svg').evaluate(svg => {
        svg.setAttribute('width', '2400');
        svg.setAttribute('height', '1760');
      });
      await page.locator('svg').screenshot({ path: source.replace(/\.svg$/, '.png'), scale: 'css' });
      console.log(source.replace(/\.svg$/, '.png'));
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exit(1); });
