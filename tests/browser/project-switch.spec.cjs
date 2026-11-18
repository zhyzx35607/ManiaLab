const { test, expect } = require('@playwright/test');

test('a late project response cannot replace the most recent selection', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option').filter({ hasText: 'UI timing clock' })).toHaveCount(1);
  await expect(page.locator('#recent option').filter({ hasText: 'UI timing wave' })).toHaveCount(1);

  const projects = await page.locator('#recent option').evaluateAll(options =>
    options.map(option => ({ id: option.value, title: option.textContent.trim() })),
  );
  const first = projects.find(project => project.title === 'UI timing clock');
  const latest = projects.find(project => project.title === 'UI timing wave');
  let releaseFirst;
  let markFirstRequested;
  const firstRequested = new Promise(resolve => { markFirstRequested = resolve; });
  const firstResponseGate = new Promise(resolve => { releaseFirst = resolve; });

  await page.route(`**/api/projects/${first.id}`, async route => {
    markFirstRequested();
    await firstResponseGate;
    await route.continue();
  });

  await page.locator('#recent').selectOption(first.id);
  await firstRequested;
  await page.locator('#recent').selectOption(latest.id);
  await expect(page.locator('#projectTitle')).toHaveText('UI timing wave');

  const firstResponse = page.waitForResponse(response =>
    response.url().endsWith(`/api/projects/${first.id}`) && response.request().method() === 'GET',
  );
  releaseFirst();
  await firstResponse;

  await expect(page.locator('#projectTitle')).toHaveText('UI timing wave');
  await expect(page.locator('#recent')).toHaveValue(latest.id);
});

test('a saved remap belongs to its difficulty after switching and reopening', async ({ page }) => {
  await page.goto('/');
  const chart = (title, keys, time) => ({
    name: `${title}.osu`, mimeType: 'text/plain',
    buffer: Buffer.from(`osu file format v14\n[General]\nMode:3\n[Metadata]\nTitle:${title}\nVersion:${title}\n[Difficulty]\nCircleSize:${keys}\n[TimingPoints]\n0,500,4,2,0,70,1,0\n[HitObjects]\n25,192,${time},128,0,${time + 705}:0:0:0:0:\n`),
  });
  await page.locator('#files').setInputFiles([chart('Chart A', 6, 1003), chart('Chart B', 9, 2100)]);
  await expect(page.locator('#projectTitle')).toHaveText('Chart A');
  await page.locator('[data-key="8"]').click();
  await page.locator('#generate').click();
  await expect(page.locator('#variants option:checked')).toContainText('8K');
  const project = await page.locator('#recent').inputValue();
  await page.locator('#charts').selectOption('Chart B.osu');
  await expect(page.locator('#projectTitle')).toHaveText('Chart B');
  await expect(page.locator('#exportOsu')).not.toHaveAttribute('href');
  await page.locator('#generate').click();
  await expect(page.locator('#variants option:checked')).toContainText('9K');
  await page.reload();
  await page.locator('#recent').selectOption(project);
  await expect(page.locator('#charts')).toHaveValue('Chart B.osu');
  await page.locator('#charts').selectOption('Chart A.osu');
  await expect(page.locator('#variants option:checked')).toContainText('8K');
  const result = await page.request.get(await page.locator('#exportOsu').getAttribute('href'));
  expect(result.ok()).toBeTruthy();
  expect(await result.text()).toContain(',1003,128,0,1708:');
});
