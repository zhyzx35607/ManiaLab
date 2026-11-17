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
