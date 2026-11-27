const { test, expect } = require('@playwright/test');

test('five music-aware presets set real traditional and Torosu preferences', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#preset option')).toHaveCount(6);
  await expect(page.locator('#singleEngine')).toHaveValue('traditional');
  await page.locator('#singleEngine').selectOption('traditional');
  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await page.locator('.right section > .advanced-settings').evaluate(node => { node.open = true; });

  await page.locator('#preset').selectOption('rice_stream');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.12');
  await expect(page.locator('#phrase_ln')).toHaveValue('0.2');
  await expect(page.locator('#layoutEngine')).toHaveValue('torosu');
  await expect(page.locator('#torosuKeepJacks')).not.toBeChecked();
  await page.locator('#preset').selectOption('rice_jack');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.12');
  await expect(page.locator('#torosuKeepJacks')).toBeChecked();
  await page.locator('#preset').selectOption('hybrid_adaptive');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.3');
  await page.locator('#preset').selectOption('ln_focus');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.55');
  await page.locator('#preset').selectOption('rice_stream');

  await page.locator('#musical > .advanced-settings').evaluate(node => { node.open = true; });
  await page.locator('#density').evaluate(control => {
    control.value = '50';
    control.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await expect(page.locator('#preset')).toHaveValue('custom');
  await expect(page.locator('#densityOut')).toHaveText('50');
  await page.locator('#timingConfirmed').check();
  let submitted;
  await page.route('**/api/projects/*/generate', async route => {
    submitted = route.request().postDataJSON();
    await route.fulfill({json:{id:'preset-generation'}});
  });
  await page.route('**/api/jobs/preset-generation', route => route.fulfill({
    json:{id:'preset-generation',status:'done',progress:100,message:'完成'},
  }));
  await page.locator('#generate').click();
  await expect.poll(()=>submitted).toBeDefined();
  expect(submitted.preset).toBeUndefined();
  expect(submitted.settings).toMatchObject({
    density:50,
    ln_ratio:0.12,
    phrase_ln:0.2,
    layout_engine:'torosu',
    torosu_keep_jacks:false,
  });
  expect(submitted.settings.ln_ratio).toBeGreaterThan(0);
});

test('an unmodified music-aware preset is submitted with its canonical id', async ({ page }) => {
  let submitted;
  await page.route('**/api/projects/*/generate', async route => {
    submitted=route.request().postDataJSON();
    await route.fulfill({json:{id:'canonical-preset-generation'}});
  });
  await page.route('**/api/jobs/canonical-preset-generation', route => route.fulfill({
    json:{id:'canonical-preset-generation',status:'done',progress:100,message:'完成'},
  }));
  await page.goto('/');
  await page.locator('#recent').selectOption({label:'UI synthetic'});
  await page.locator('#timingConfirmed').check();
  await page.locator('#preset').selectOption('rice_stream');
  await page.locator('#generate').click();
  await expect.poll(()=>submitted).toBeDefined();
  expect(submitted.preset).toBe('rice_stream');
  expect(submitted.settings).toMatchObject({
    ln_ratio:0.12,
    phrase_ln:0.2,
    layout_engine:'torosu',
    torosu_keep_jacks:false,
  });
});

test('saved empty projects keep their settings while newly imported audio gets the hybrid default', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option').filter({ hasText: 'UI synthetic' })).toHaveCount(1);
  const projectId = await page.locator('#recent option').evaluateAll(options =>
    options.find(option => option.textContent === 'UI synthetic').value);
  await page.locator('#recent').selectOption(projectId);
  const saved = await page.evaluate(async id => {
    const response = await fetch(`/api/projects/${id}`);
    return response.json();
  }, projectId);
  await expect(page.locator('#layoutEngine')).toHaveValue(saved.settings.layout_engine);
  await expect(page.locator('#ln_ratio')).toHaveValue(String(saved.settings.ln_ratio));

  const imported = structuredClone(saved);
  imported.title = 'Fresh audio import';
  imported.variants = [];
  imported.active = -1;
  await page.route('**/api/import', route => route.fulfill({ json: imported }));
  await page.locator('#files').setInputFiles([{
    name: 'fresh-audio.wav',
    mimeType: 'audio/wav',
    buffer: Buffer.from('mock import payload'),
  }]);

  await expect(page.locator('#projectTitle')).toHaveText('Fresh audio import');
  await expect(page.locator('#preset')).toHaveValue('hybrid_adaptive');
  await expect(page.locator('#layoutEngine')).toHaveValue('torosu');
  await expect(page.locator('#ln_ratio')).toHaveValue('0.3');
});
