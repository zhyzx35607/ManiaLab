const { test, expect } = require('@playwright/test');

test('single controls disable without a project and disappear in batch mode', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#generate')).toBeDisabled();
  await expect(page.locator('#another')).toBeDisabled();
  await page.locator('[data-work-mode="batch"]').click();
  await expect(page.locator('.right')).toBeHidden();
  await expect(page.locator('#batchDownloadPack')).toBeHidden();
  await page.locator('[data-work-mode="single"]').click();
  await expect(page.locator('.right')).toBeVisible();
});

test('V32 custom tuning remains generatable and legacy controls are isolated', async ({ page }) => {
  const response = await page.request.get('/api/status');
  const status = await response.json();
  status.v32_engine = { ...status.v32_engine, available: true, runtime_available: true,
    device: { type: 'cuda', name: 'RTX 4050' }, models: { 'v32-mini': { ready: true }, v32: { ready: true } } };
  status.projects = status.projects.filter(project => project.title === 'UI synthetic');
  await page.route('**/api/status', route => route.fulfill({ json: status }));
  await page.goto('/');
  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await expect(page.locator('#generate')).toBeEnabled();
  await page.locator('#singleEngine').selectOption('traditional');
  await page.locator('#singleTraditionalSettings').evaluate(element => { element.open = true; });
  await page.locator('#density').evaluate(element => {
    element.value = '9'; element.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(page.locator('#preset')).toHaveValue('custom');
  await page.locator('#singleEngine').selectOption('v32-mini');
  await expect(page.locator('#singleTraditionalSettings')).toBeHidden();
  await page.locator('.v32-panel').evaluate(element => { element.open = true; });
  await page.locator('#v32Density').selectOption('more-dense');
  await page.locator('#v32Difficulty').fill('6.2');
  let submitted;
  await page.route(/\/api\/projects\/[^/]+\/v32-generate$/, async route => {
    submitted = route.request().postDataJSON();
    await route.fulfill({ json: { id: 'custom-v32-job' } });
  });
  await page.route('**/api/jobs/custom-v32-job', route => route.fulfill({ json: {
    id: 'custom-v32-job', status: 'done', progress: 100, message: '完成',
  } }));
  await page.locator('#generate').click();
  await expect.poll(() => submitted).toBeDefined();
  expect(submitted).toMatchObject({ preset: 'hybrid_adaptive', torosu_density: 'more-dense', difficulty: 6.2 });
  await page.locator('#singleEngine').selectOption('traditional');
  await expect(page.locator('#singleTraditionalSettings')).toBeVisible();
});

test('metadata editor saves export fields through the project API', async ({ page }) => {
  await page.goto('/');
  const imported = page.waitForResponse(response => response.url().endsWith('/api/import') && response.request().method() === 'POST');
  await page.locator('#files').setInputFiles({ name: 'beta3-metadata.wav', mimeType: 'audio/wav',
    buffer: Buffer.from('offline metadata fixture') });
  await expect(page.locator('#projectTitle')).toHaveText('beta3-metadata');
  const projectId = (await (await imported).json()).id;
  const original = await (await page.request.get(`/api/projects/${projectId}/metadata`)).json();
  await expect(page.locator('#metadataEditor')).toBeVisible();
  await page.locator('#metadataEditor summary').click();
  await page.locator('[name="title"]').fill('Mapper Workflow');
  await page.locator('[name="title_unicode"]').fill('beta3-metadata');
  await page.locator('[name="creator"]').fill('Mapper');
  await page.locator('[name="version"]').fill('Hybrid');
  await page.locator('[name="od"]').fill('8');
  await page.locator('#metadataSave').click();
  await expect(page.locator('#message')).toContainText('Metadata 已保存');
  const response = await page.request.get(`/api/projects/${projectId}/metadata`);
  expect(await response.json()).toMatchObject({
    Title: 'Mapper Workflow', TitleUnicode: 'beta3-metadata', Creator: 'Mapper', Version: 'Hybrid', OverallDifficulty: 8,
  });
});

test('batch mode exposes the primary pack and the legacy independent bundle separately', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  const primary = page.locator('#batchDownloadPack');
  const alternate = page.locator('#batchDownloadAll');
  await expect(primary).toHaveText(/合集包（单个 \.osz）/);
  await expect(alternate).toHaveText(/独立谱包（ZIP）/);
  await expect(primary).toHaveAttribute('aria-disabled', 'true');
  await expect(alternate).toHaveAttribute('aria-disabled', 'true');
});
