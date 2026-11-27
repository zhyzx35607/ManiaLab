const { test, expect } = require('@playwright/test');

const v32Status = {
  available: true,
  source_available: true,
  runtime_available: true,
  device: {
    type: 'cuda',
    name: 'RTX 4050',
    memory_gib: 6,
    precision: 'bf16',
    fast_decoder_loop: true,
  },
  models: {
    v32: { ready: true, revision: '74f22583400d' },
    'v32-mini': { ready: true, revision: '7807f0dc70ca' },
  },
  lora: { loaded: false, path: null },
  separation_available: true,
};

async function openV32Project(page, { canInstall = false, separationAvailable = true } = {}) {
  const response = await page.request.get('/api/status');
  const status = await response.json();
  status.v32_engine = {
    ...v32Status,
    available: !canInstall,
    source_available: !canInstall,
    runtime_available: !canInstall,
    install_available: canInstall,
    separation_available: separationAvailable,
  };
  await page.route('**/api/status', route => {
    return route.fulfill({ json: status });
  });
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI synthetic']);
  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await expect(page.locator('#projectTitle')).toHaveText('UI synthetic');
  await page.locator('#preset').selectOption('hybrid_adaptive');
  await page.locator('.v32-panel').evaluate(node => { node.open = true; });
  await expect(page.locator('#v32Status')).toContainText('RTX 4050');
  if (canInstall) {
    await expect(page.locator('#v32Install')).toBeVisible();
  } else {
    await expect(page.locator('#v32Generate')).toBeEnabled();
  }
  return page.locator('#recent').inputValue();
}

test('V32 controls submit the selected model and generation settings through the task queue', async ({ page }) => {
  const projectId = await openV32Project(page);
  await page.locator('#timingConfirmed').uncheck();
  let submitted;
  await page.route(`**/api/projects/${projectId}/v32-generate`, async route => {
    submitted = route.request().postDataJSON();
    await route.fulfill({ json: { id: 'v32-ui-job' } });
  });
  await page.route('**/api/jobs/v32-ui-job', route => route.fulfill({
    json: { id: 'v32-ui-job', status: 'done', progress: 100, message: '完成' },
  }));
  await page.locator('#v32Model').selectOption('v32-mini');
  await page.locator('#v32Strategy').selectOption('native');
  await page.locator('#v32Density').selectOption('more-dense');
  await page.locator('#v32Difficulty').fill('6.2');
  await page.locator('#v32HoldRatio').fill('0.65');
  await page.locator('#v32Temperature').fill('1.3');
  await page.locator('#v32LoraPath').fill('models/10k-lora');

  await page.locator('#v32Generate').click();

  await expect.poll(() => submitted).toBeDefined();
  await expect.poll(() => submitted).toEqual({
    preset: 'hybrid_adaptive',
    keys: 10,
    key_layout: '10k_5_5',
    bpm: 120,
    offset: 0,
    division: 4,
    audio_shift_ms: 0,
    model_variant: 'v32-mini',
    generation_strategy: 'native',
    torosu_density: 'more-dense',
    torosu_keep_jacks: true,
    torosu_ln_mode: 'keep',
    difficulty: 6.2,
    hold_note_ratio: 0.65,
    mania_column_temperature: 1.3,
    lora_path: 'models/10k-lora',
    metadata: { creator: 'zhyzx35607', version_suffix: '' },
    timing_confirmed: false,
  });
  await expect(page.locator('#job')).toHaveClass(/hidden/);
});

test('lightweight workbench keeps V32 available through the mixed-audio RMS fallback', async ({ page }) => {
  const projectId = await openV32Project(page, { separationAvailable: false });
  await expect(page.locator('#v32Generate')).toBeEnabled();
  await expect(page.locator('#v32Status')).toContainText('原混音 RMS fallback');
  await expect(page.locator('#v32ReadyStatus')).toContainText('混音 RMS fallback');

  let submitted;
  await page.route(`**/api/projects/${projectId}/v32-generate`, async route => {
    submitted = route.request().postDataJSON();
    await route.fulfill({ json: { id: 'v32-rms-fallback-job' } });
  });
  await page.route('**/api/jobs/v32-rms-fallback-job', route => route.fulfill({
    json: { id: 'v32-rms-fallback-job', status: 'done', progress: 100, message: '完成' },
  }));
  await page.locator('#v32Generate').click();
  await expect.poll(() => submitted).toBeDefined();
  expect(submitted).toMatchObject({
    model_variant: 'v32-mini',
    preset: 'hybrid_adaptive',
  });
});

test('V32 queued jobs use the shared cancellation endpoint', async ({ page }) => {
  await openV32Project(page);
  let cancelled = false;
  await page.route('**/api/projects/*/v32-generate', route => route.fulfill({
    json: { id: 'v32-cancel-job' },
  }));
  await page.route('**/api/jobs/v32-cancel-job/cancel', async route => {
    cancelled = true;
    await route.fulfill({ json: { ok: true } });
  });
  await page.route('**/api/jobs/v32-cancel-job', route => route.fulfill({
    json: {
      id: 'v32-cancel-job',
      status: cancelled ? 'cancelled' : 'running',
      progress: 25,
      message: cancelled ? '已取消，原有候选已保留' : '正在 V32 推理',
    },
  }));
  await page.locator('#v32Generate').click();
  await expect(page.locator('#cancel')).toBeVisible();
  await page.locator('#cancel').click();
  await expect.poll(() => cancelled).toBe(true);
  await expect(page.locator('#message')).toContainText('已取消');
});

test('V32 first-run setup runs through the shared cancellable task queue', async ({ page }) => {
  await openV32Project(page, { canInstall: true });
  let submitted;
  await page.route('**/api/v32/install', route => {
    submitted = route.request().postDataJSON();
    return route.fulfill({ json: { id: 'v32-install-job', global: true } });
  });
  await page.route('**/api/jobs/v32-install-job', route => route.fulfill({
    json: {
      id: 'v32-install-job',
      status: 'done',
      progress: 100,
      message: 'V32 推理环境已就绪',
    },
  }));
  await page.locator('#v32Install').click();
  await expect.poll(() => submitted).toEqual({});
  await expect(page.locator('#job')).toHaveClass(/hidden/);
});

test('V32 setup is available globally before a project is open', async ({ page }) => {
  const response = await page.request.get('/api/status');
  const status = await response.json();
  status.v32_engine = {
    ...v32Status,
    available: false,
    source_available: true,
    runtime_available: false,
    install_available: true,
    separation_available: false,
  };
  let installRequested = false;
  await page.route('**/api/status', route => route.fulfill({ json: status }));
  await page.route('**/api/v32/install', async route => {
    installRequested = true;
    await route.fulfill({ json: { id: 'v32-global-install-job', global: true } });
  });
  await page.route('**/api/jobs/v32-global-install-job', route => route.fulfill({
    json: {
      id: 'v32-global-install-job',
      status: 'done',
      progress: 100,
      message: 'V32 推理环境已就绪',
    },
  }));

  await page.goto('/');
  await expect(page.locator('#recent')).toHaveValue('');
  await expect(page.locator('#headerV32Install')).toBeVisible();
  await expect(page.locator('#headerV32Install')).toBeEnabled();
  await page.locator('#headerV32Install').click();

  await expect.poll(() => installRequested).toBe(true);
  await expect(page.locator('#message')).toContainText('V32 全局运行环境已准备完成');
});
