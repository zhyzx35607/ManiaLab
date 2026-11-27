const { test, expect } = require('@playwright/test');

function wavBase64(seconds = 8) {
  const sampleRate = 8000;
  const bytesPerSample = 2;
  const sampleCount = sampleRate * seconds;
  const dataBytes = sampleCount * bytesPerSample;
  const wav = Buffer.alloc(44 + dataBytes);
  wav.write('RIFF', 0);
  wav.writeUInt32LE(36 + dataBytes, 4);
  wav.write('WAVE', 8);
  wav.write('fmt ', 12);
  wav.writeUInt32LE(16, 16);
  wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(sampleRate * bytesPerSample, 28);
  wav.writeUInt16LE(bytesPerSample, 32);
  wav.writeUInt16LE(16, 34);
  wav.write('data', 36);
  wav.writeUInt32LE(dataBytes, 40);
  return wav.toString('base64');
}

async function mountPreview(page, keys) {
  await page.goto('/');
  await page.evaluate(() => {
    document.querySelector('#chartPreviewHost')?.remove();
    const root = document.createElement('div');
    root.id = 'preview';
    const audio = document.createElement('audio');
    audio.id = 'source';
    audio.muted = true;
    audio.preload = 'auto';
    document.querySelector('article').append(root, audio);
  });
  await page.evaluate(async ({ keysValue, wav }) => {
    const audio = document.querySelector('#source');
    const binary = atob(wav);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    audio.src = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
    audio.load();
    if (audio.readyState < 1) {
      await new Promise((resolve, reject) => {
        audio.addEventListener('loadedmetadata', resolve, { once: true });
        audio.addEventListener('error', reject, { once: true });
      });
    }
    const preview = window.ManiaLabChartPreview.mount(
      document.querySelector('#preview'),
      audio,
    );
    window.testPreview = preview;
    window.testAudio = audio;
    preview.setChart({
      keys: keysValue,
      bpm: 120,
      offset: 0,
      division: 4,
      notes: [
        { start_ms: 800, end_ms: 800, lane: 0 },
        { start_ms: 1200, end_ms: 2600, lane: Math.floor(keysValue / 2), is_hold: true },
        { start_ms: 3200, end_ms: 3200, lane: keysValue - 1 },
      ],
      stats: {
        peak_held: 1,
        torosu: {
          minijack: 3,
          pill: 4,
          hand_balance: 0.94,
          pinky_balance: 0.82,
          lane_distribution: Array.from({ length: keysValue }, (_, index) => index + 2),
        },
      },
    });
  }, { keysValue: keys, wav: wavBase64() });
}

test('6K and 10K vertical previews render taps and LN bodies from the media clock', async ({ page }) => {
  for (const keys of [6, 10]) {
    await mountPreview(page, keys);
    const canvas = page.locator('[data-preview-canvas]');
    await expect(page.locator('#preview style')).toHaveCount(0);
    await expect(canvas).toHaveAttribute('aria-label', '竖屏下落式谱面预览');
    await expect(page.locator('[data-preview-stats]')).toContainText('3');
    await expect(page.locator('[data-preview-stats]')).toContainText('33.3%');
    await expect(page.locator('[data-preview-torosu]')).toContainText('Minijack 3');
    await expect(page.locator('[data-preview-torosu]')).toContainText('Pill 4');

    const paintedPixels = await page.evaluate(() => {
      const canvasElement = document.querySelector('[data-preview-canvas]');
      const context = canvasElement.getContext('2d');
      const pixels = context.getImageData(0, 0, canvasElement.width, canvasElement.height).data;
      const colors = new Set();
      let nonBackground = 0;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        colors.add(`${pixels[offset]},${pixels[offset + 1]},${pixels[offset + 2]}`);
        if (pixels[offset] !== 16 || pixels[offset + 1] !== 26 || pixels[offset + 2] !== 40) {
          nonBackground += 1;
        }
      }
      return { colors: colors.size, nonBackground };
    });
    expect(paintedPixels.colors).toBeGreaterThan(10);
    expect(paintedPixels.nonBackground).toBeGreaterThan(1000);

    const holdImage = await page.evaluate(() => {
      const canvasElement = document.querySelector('[data-preview-canvas]');
      return canvasElement.toDataURL();
    });
    await page.evaluate(keysValue => {
      window.testPreview.setChart({
        keys: keysValue,
        bpm: 120,
        notes: [
          { start_ms: 800, end_ms: 800, lane: 0 },
          { start_ms: 3200, end_ms: 3200, lane: keysValue - 1 },
        ],
      });
    }, keys);
    const tapImage = await page.evaluate(() => document
      .querySelector('[data-preview-canvas]')
      .toDataURL());
    expect(holdImage).not.toBe(tapImage);

    await page.locator('[data-preview-progress]').fill('2.5');
    await expect.poll(() => page.evaluate(() => window.testAudio.currentTime)).toBeCloseTo(2.5, 1);
    await page.locator('[data-preview-rate]').fill('0.5');
    await expect.poll(() => page.evaluate(() => window.testAudio.playbackRate)).toBeCloseTo(0.5, 1);
    await page.locator('[data-preview-rate]').fill('1.5');
    await expect.poll(() => page.evaluate(() => window.testAudio.playbackRate)).toBeCloseTo(1.5, 1);
    await page.locator('[data-preview-scroll]').fill('0.5');
    await expect(page.locator('[data-preview-scroll-value]')).toHaveText('0.5x');
    await page.locator('[data-preview-scroll]').fill('1.4');
    await expect(page.locator('[data-preview-scroll-value]')).toHaveText('1.4x');

    await page.locator('[data-preview-play]').click();
    await expect(page.locator('[data-preview-play]')).toHaveAttribute('aria-label', '暂停预览');
    await expect.poll(() => page.evaluate(() => window.testAudio.currentTime)).toBeGreaterThan(2.7);
    await page.locator('[data-preview-play]').click();
    await expect(page.locator('[data-preview-play]')).toHaveAttribute('aria-label', '播放预览');

    await page.evaluate(() => window.testPreview.destroy());
  }
});

test('preview accepts an empty chart and rejects unsupported key counts', async ({ page }) => {
  await mountPreview(page, 7);
  await page.evaluate(() => window.testPreview.setChart({
    notes: [], keys: 7, bpm: 120, stats: {},
  }));
  await expect(page.locator('[data-preview-empty]')).toBeVisible();
  await expect(page.locator('[data-preview-stats]')).toContainText('0');
  await expect(page.locator('[data-preview-torosu]')).toContainText('尚未提供');
  const errorMessage = await page.evaluate(() => {
    try {
      window.testPreview.setChart({ notes: [], keys: 11 });
    } catch (error) {
      return error.message;
    }
    return '';
  });
  expect(errorMessage).toContain('between 6 and 10');
  await page.evaluate(() => window.testPreview.destroy());
});
