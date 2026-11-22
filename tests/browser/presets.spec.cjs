const { test, expect } = require('@playwright/test');

test('onset-only preset sends zero LN ratios and generates only taps', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#preset')).toBeVisible();
  await expect(page.locator('#generate')).toBeVisible();
  await expect(page.locator('#musical > .advanced-settings')).not.toHaveAttribute('open', '');
  await expect(page.locator('.right section > .advanced-settings')).not.toHaveAttribute('open', '');
  await expect(page.locator('#lnDesignPanel')).not.toHaveAttribute('open', '');

  await page.locator('.right section > .advanced-settings').evaluate(node => { node.open = true; });
  await expect(page.locator('.ln-preferences')).not.toHaveAttribute('open', '');

  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await page.locator('#preset').selectOption('onset_only');
  await expect(page.locator('#ln_ratio')).toHaveValue('0');
  await expect(page.locator('#phrase_ln')).toHaveValue('0');
  await page.locator('#preset').selectOption('balanced');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.55');
  await expect(page.locator('#phrase_ln')).toHaveValue('0.65');
  await page.locator('#preset').selectOption('onset_only');
  await page.locator('#timingConfirmed').check();

  const generation = page.waitForRequest(request =>
    request.url().endsWith('/generate') && request.method() === 'POST');
  await page.locator('#generate').click();
  const request = await generation;
  expect(request.postDataJSON().settings).toMatchObject({ ln_ratio: 0, phrase_ln: 0 });
  await expect(page.locator('#message')).toContainText('结构检查通过');

  const projectId = await page.locator('#recent').inputValue();
  const project = await (await page.request.get(`/api/projects/${projectId}`)).json();
  const chart = project.variants[project.active];
  expect(chart.report.ln).toBe(0);
  expect(chart.notes.length).toBeGreaterThan(0);
  expect(chart.notes.every(note => note.start === note.end)).toBeTruthy();
});
