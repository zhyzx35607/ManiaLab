const { test, expect } = require('@playwright/test');

function project(id, index) {
  return {
    id,
    title: `LN Batch Song ${index + 1}`,
    artist: 'Batch fixture',
    mode: 'audio',
    settings: { keys: 10 },
    variants: [],
    active: -1,
    charts: [],
    audio: `${id}.wav`,
    original: null,
    input_notes: [],
    audio_shift_ms: 22.5,
    analysis: {
      revision: 3,
      tempo: { bpm: 120, offset: 93, division: 4, stable: true },
    },
  };
}

const audioFile = index => ({
  name: `v32-song-${index + 1}.wav`,
  mimeType: 'audio/wav',
  buffer: Buffer.from(`synthetic batch fixture ${index}`),
});

test('20-song V32 batch forwards one preset and exports all completed candidates', async ({ page }) => {
  const ids = Array.from({ length: 20 }, (_, index) => `v32-batch-${index + 1}`);
  const states = new Map(ids.map((id, index) => [id, project(id, index)]));
  const candidateIds = ids.map((_, index) => (index + 1).toString(16).padStart(32, '0'));
  let analysisRequest;
  let generationRequest;
  let exportedItems;

  const status = await (await page.request.get('/api/status')).json();
  status.v32_engine = {
    ...(status.v32_engine || {}),
    available: true,
    source_available: true,
    runtime_available: true,
    device: { type: 'cuda', name: 'RTX 4050', memory_gib: 6, precision: 'bf16' },
    models: {
      v32: { ready: true, revision: 'phase06-test' },
      'v32-mini': { ready: true, revision: 'phase06-test' },
    },
    lora: { loaded: false, path: null },
  };
  await page.route('**/api/status', route => route.fulfill({ json: status }));
  await page.route('**/api/loras', route => route.fulfill({ json: {
    items: [{ id: 'ln-adapter', label: 'LN Adapter', path: 'assets/lora/ln-adapter', base_model: 'v32' }],
  } }));
  await page.route('**/api/import', async route => {
    const body = route.request().postData() || '';
    const index = ids.findIndex((_, row) => body.includes(`v32-song-${row + 1}.wav`));
    await route.fulfill({ json: project(ids[index], index) });
  });
  await page.route('**/api/batch/analyze', async route => {
    analysisRequest = route.request().postDataJSON();
    await route.fulfill({ json: {
      id: 'v32-batch-analysis',
      status: 'queued',
      batch: true,
      progress: 0,
      items: ids.map((id, index) => ({
        project: id,
        title: states.get(id).title,
        status: 'queued',
        progress: 0,
      })),
    } });
  });
  await page.route('**/api/jobs/v32-batch-analysis', route => route.fulfill({ json: {
    id: 'v32-batch-analysis',
    status: 'done',
    batch: true,
    progress: 100,
    items: ids.map((id, index) => ({
      project: id,
      title: states.get(id).title,
      status: 'done',
      progress: 100,
      result: { candidates: index + 1 },
    })),
  } }));
  await page.route(/\/api\/projects\/v32-batch-\d+$/, async route => {
    const id = new URL(route.request().url()).pathname.split('/').at(-1);
    await route.fulfill({ json: states.get(id) });
  });
  await page.route('**/api/batch/generate', async route => {
    generationRequest = route.request().postDataJSON();
    await route.fulfill({ json: {
      id: 'v32-batch-generation',
      status: 'queued',
      batch: true,
      progress: 0,
      generation_total: ids.length,
      items: ids.map((id, index) => ({
        project: id,
        title: states.get(id).title,
        status: 'queued',
        progress: 0,
      })),
    } });
  });
  await page.route('**/api/jobs/v32-batch-generation', route => route.fulfill({ json: {
    id: 'v32-batch-generation',
    status: 'done',
    batch: true,
    progress: 100,
    generation_total: ids.length,
    items: ids.map((id, index) => ({
      project: id,
      title: states.get(id).title,
      status: 'done',
      progress: 100,
      result: { candidate_id: candidateIds[index], report: { ln: 3 } },
    })),
  } }));
  await page.route('**/api/batch/export-osz?*', async route => {
    exportedItems = new URL(route.request().url()).searchParams.get('items');
    await route.fulfill({
      status: 200,
      headers: {
        'content-type': 'application/zip',
        'content-disposition': 'attachment; filename="ManiaLab-batch-osz.zip"',
      },
      body: Buffer.from('PK\x03\x04'),
    });
  });

  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  await page.locator('#batchFiles').setInputFiles(ids.map((_, index) => audioFile(index)));
  await expect(page.locator('#batchFileSummary')).toContainText('20 首');
  await expect(page.locator('#batchAnalyze')).toBeEnabled();
  await page.locator('#batchAnalyze').click();
  await expect(page.locator('#batchStatus')).toContainText('批次完成：20/20 首成功');
  expect(analysisRequest.projects).toEqual(ids);

  await page.locator('#batchKeys').selectOption('9');
  await page.locator('#batchEngine').selectOption('v32');
  await page.locator('#batchPreset').selectOption('ln_focus');
  await page.locator('#batchTorosuDensity').selectOption('full');
  await page.locator('#batchTorosuLnMode').selectOption('short_ln');
  await page.locator('#batchDifficulty').fill('7.2');
  await page.locator('#batchHoldRatio').fill('0.7');
  await page.locator('#batchLora').selectOption('assets/lora/ln-adapter');
  await page.locator('#batchScribe').fill('Mapper Nine');
  await page.locator('#batchVersionSuffix').fill('ln-v2');
  await page.locator('#batchGenerate').click();

  await expect(page.locator('#batchStatus')).toContainText('统一生成结束：20/20 首成功');
  expect(generationRequest).toMatchObject({
    projects: ids,
    preset: 'ln_focus',
    engine: 'v32',
    model_variant: 'v32',
    keys: 9,
    key_layout: '9k_4_1_4',
    torosu_density: 'full',
    torosu_ln_mode: 'short_ln',
    difficulty: 7.2,
    hold_note_ratio: 0.7,
    lora_path: 'assets/lora/ln-adapter',
    scribe: 'Mapper Nine',
    version_suffix: 'ln-v2',
  });
  await expect(page.locator('#batchStatus [data-batch-download]')).toHaveCount(20);
  await expect(page.locator('#batchDownloadAll')).toHaveAttribute('href', /\/api\/batch\/export-osz\?/);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#batchDownloadAll').click();
  expect(await (await downloadPromise).failure()).toBeNull();
  expect(exportedItems.split(',')).toHaveLength(20);
  expect(exportedItems.split(',')[0]).toBe(`${ids[0]}:${candidateIds[0]}`);
  expect(exportedItems.split(',')[19]).toBe(`${ids[19]}:${candidateIds[19]}`);
});

test('a 21-song file selection is refused before batch import', async ({ page }) => {
  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  await page.locator('#batchFiles').setInputFiles(
    Array.from({ length: 21 }, (_, index) => audioFile(index)),
  );
  await expect(page.locator('#batchFileSummary')).toContainText('最多处理 20 首');
  await expect(page.locator('#batchAnalyze')).toBeDisabled();
});
