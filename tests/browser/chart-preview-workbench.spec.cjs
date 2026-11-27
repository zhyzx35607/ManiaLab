const { test, expect } = require('@playwright/test');

test('the real workbench preview stays inside desktop and mobile viewports', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  const response = await page.goto('/');
  expect(response.headers()['content-security-policy']).toContain("style-src 'self'");
  await expect(page.locator('#recent option')).toContainText(['UI synthetic']);
  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');

  const canvas = page.locator('#chartPreviewHost [data-preview-canvas]');
  await page.locator('#fallingPreviewPanel > summary').click();
  await expect(canvas).toBeVisible();
  await expect(page.locator('#chartPreviewHost style')).toHaveCount(0);
  await expect(page.locator('#chartPreviewHost [data-preview-stats]')).toContainText('物件');
  await expect(page.locator('#chartPreviewHost [data-preview-empty]')).toBeHidden();

  for (const viewport of [
    { width: 1440, height: 1000 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    const dimensions = await page.evaluate(() => {
      const canvasElement = document.querySelector('#chartPreviewHost [data-preview-canvas]');
      const stage = document.querySelector('#chartPreviewHost .ml-preview-stage');
      const host = document.querySelector('#chartPreviewHost');
      return {
        viewport: document.documentElement.clientWidth,
        document: document.documentElement.scrollWidth,
        host: host.getBoundingClientRect().width,
        stage: {
          width: stage.getBoundingClientRect().width,
          height: stage.getBoundingClientRect().height,
        },
        canvas: canvasElement.getBoundingClientRect().width,
        painted: canvasElement.getContext('2d').getImageData(
          0,
          0,
          canvasElement.width,
          canvasElement.height,
        ).data.some((value, index) => index % 4 !== 3 && value !== 0),
      };
    });
    expect(dimensions.document).toBeLessThanOrEqual(dimensions.viewport + 1);
    expect(dimensions.stage.width).toBeLessThanOrEqual(dimensions.host + 1);
    expect(dimensions.stage.height).toBeGreaterThan(dimensions.stage.width);
    expect(dimensions.stage.height / dimensions.stage.width).toBeCloseTo(16 / 9, 1);
    expect(dimensions.canvas).toBeLessThanOrEqual(dimensions.stage.width + 1);
    expect(dimensions.painted).toBe(true);
  }
});
