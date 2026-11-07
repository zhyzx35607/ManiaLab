// Inspect a generated long project through the real saved-project selector.
const {chromium} = require('../test-data/ui/node_modules/playwright');
const fs = require('fs');
const path = require('path');

(async () => {
  const base = process.env.MANIALAB_URL;
  const project = process.env.MANIALAB_TEST_PROJECT;
  if (!base || !project) throw Error('Set MANIALAB_URL and MANIALAB_TEST_PROJECT');
  const out = process.env.MANIALAB_TEST_OUTPUT || 'test-data/long-duration-ui';
  fs.mkdirSync(out, {recursive: true});
  const browser = await chromium.launch({headless: true, executablePath: 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
  try {
    const page = await browser.newPage({viewport: {width: 1512, height: 1100}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(base);
    await page.waitForFunction(id => [...document.querySelector('#recent').options].some(o => o.value === id), project);
    const start = Date.now();
    await page.locator('#recent').selectOption(project);
    await page.waitForFunction(id => state?.id === id && !!variant(), project, {timeout: 60000});
    const loadedMs = Date.now() - start;
    const counts = await page.evaluate(() => ({duration: duration(), notes: variant().notes.length, candidates: state.analysis.candidates.length}));
    if (counts.duration < 1199 || counts.notes < 5000) throw Error('Expected a twenty-minute generated project');
    const timings = [];
    for (const seconds of [1, 600, 1190]) {
      await page.locator('#seek').fill(String(seconds));
      await page.locator('#seek').dispatchEvent('input');
      for (const mode of ['chart', 'evidence']) {
        await page.locator('#' + mode + 'View').click();
        timings.push(await page.evaluate(({seconds, mode}) => {
          const begin = performance.now();
          draw();
          return {seconds, mode, drawMs: performance.now() - begin, visible: hitboxes.length, displayed: displayed.length};
        }, {seconds, mode}));
      }
    }
    await page.locator('#chartView').click();
    await page.locator('#waveDetailPanel summary').click();
    await page.locator('#waveHere').click();
    await page.waitForFunction(() => !!waveDetail, {}, {timeout: 30000});
    const wave = await page.evaluate(() => ({start: waveDetail.start_ms, end: waveDetail.end_ms}));
    if (wave.start < 1180000) throw Error('Long-project waveform seek used the wrong time');
    await page.screenshot({path: path.join(out, 'long-project-desktop.png'), fullPage: true});
    const downloadPromise = page.waitForEvent('download');
    await page.locator('#exportOsu').click();
    const download = await downloadPromise;
    await download.saveAs(path.join(out, 'long-project.osu'));
    await page.setViewportSize({width: 390, height: 844});
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    const report = {counts, loadedMs, timings, wave, errors, overflow,
      note: 'Observed browser timings on one host, not a performance guarantee. Source is an original repeated synthetic fixture.'};
    fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report));
    if (errors.length || overflow) process.exitCode = 1;
  } finally {
    await browser.close();
  }
})().catch(e => {console.error(e); process.exit(1);});
