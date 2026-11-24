const { test, expect } = require('@playwright/test');

function project(id, title) {
  return {
    id, title, artist: 'Synthetic', mode: 'audio',
    settings: { keys: 10 }, variants: [], active: -1, charts: [],
    audio: `${id}.wav`, original: null, input_notes: [], audio_shift_ms: 22.5,
    analysis: null,
  };
}

const audioFile = name => ({
  name, mimeType: 'audio/wav', buffer: Buffer.from('synthetic fixture audio'),
});

test('batch analysis imports ten songs, caps parallelism, and isolates a failed song', async ({ page }) => {
  const imported = [];
  let batchRequest;
  let pollCount = 0;
  const ids = Array.from({ length: 10 }, (_, index) => `batch-song-${index + 1}`);
  const titles = Array.from({ length: 10 }, (_, index) => `Song ${index + 1}`);

  await page.route('**/api/import', async route => {
    const index = imported.length;
    imported.push(route.request().postData() || '');
    await route.fulfill({ json: project(ids[index], titles[index]) });
  });
  await page.route('**/api/batch/analyze', async route => {
    batchRequest = JSON.parse(route.request().postData() || '{}');
    await route.fulfill({ json: {
      id: 'batch-fixture', status: 'queued', batch: true, progress: 0,
      message: '等待批量分析',
      items: ids.map((id, index) => ({
        project: id, title: titles[index], status: 'queued', progress: 0, message: '等待运行',
      })),
    } });
  });
  await page.route('**/api/jobs/batch-fixture', async route => {
    pollCount += 1;
    const finished = pollCount >= 2;
    const items = ids.map((id, index) => {
      if (index === 1) return { project: id, title: titles[index], status: 'error', progress: 100, message: '合成分析失败' };
      const running = index === 9 && !finished;
      return {
        project: id, title: titles[index], status: running ? 'running' : 'done',
        progress: running ? 60 : 100, message: running ? '正在提取音符' : '完成',
        ...(!running ? { result: { candidates: index + 1 } } : {}),
      };
    });
    await route.fulfill({ json: {
      id: 'batch-fixture', status: finished ? 'done' : 'running', batch: true,
      progress: finished ? 100 : 95, message: finished ? '批量分析完成' : '正在分析 9 / 10', items,
    } });
  });
  await page.route(/\/api\/projects\/batch-song-1$/, async route => {
    await route.fulfill({ json: project(ids[0], titles[0]) });
  });
  await page.route('**/api/projects/batch-song-1/audio*', route => route.fulfill({ status: 404, body: '' }));

  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles(
    ids.map((_, index) => audioFile(`song-${index + 1}.wav`)),
  );
  await page.locator('#batchAnalyze').click();

  await expect(page.locator('#batchStatus [data-batch-item]')).toHaveCount(10);
  for (const title of titles) await expect(page.locator('#batchStatus')).toContainText(title);
  await expect(page.locator('#batchStatus')).toContainText('完成');
  await expect(page.locator('#batchStatus')).toContainText('失败');

  expect(imported).toHaveLength(10);
  expect(imported[0]).toContain('song-1.wav');
  expect(imported[9]).toContain('song-10.wav');
  expect(batchRequest.projects).toEqual(ids);
  expect(batchRequest.max_parallel).toBe(2);

  await page.locator('[data-open-project="batch-song-1"]').click();
  await expect(page.locator('#projectTitle')).toHaveText('Song 1');
});

test('selecting eleven songs disables batch analysis', async ({ page }) => {
  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles(
    Array.from({ length: 11 }, (_, index) => audioFile(`song-${index + 1}.wav`)),
  );

  await expect(page.locator('#batchAnalyze')).toBeDisabled();
  await expect(page.locator('#batchFileSummary')).toContainText('10');
});

test('cancelling while an import is in flight prevents the next song from starting', async ({ page }) => {
  let importCount = 0;
  let releaseImport;
  let batchStarted = false;
  await page.route('**/api/import', async route => {
    importCount += 1;
    await new Promise(resolve => { releaseImport = () => resolve(route.fulfill({ json: project('cancel-song-1', 'Cancel Song One') })); });
  });
  await page.route('**/api/batch/analyze', async route => {
    batchStarted = true;
    await route.fulfill({ json: { id: 'should-not-start', status: 'queued', items: [] } });
  });

  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles([
    audioFile('cancel-one.wav'), audioFile('cancel-two.wav'),
  ]);
  await page.locator('#batchAnalyze').click();
  await expect.poll(() => importCount).toBe(1);
  await page.locator('#cancel').click();
  await expect(page.locator('#jobText')).toContainText('正在取消批量导入');
  releaseImport();

  await expect(page.locator('#message')).toContainText('批量导入已取消');
  await expect(page.locator('#batchStatus')).toContainText('Cancel Song One');
  expect(importCount).toBe(1);
  expect(batchStarted).toBe(false);
});

test('a temporary polling error is retried until the batch result returns', async ({ page }) => {
  const ids = ['retry-song-1', 'retry-song-2'];
  const titles = ['Retry Song One', 'Retry Song Two'];
  let pollCount = 0;
  await page.route('**/api/import', async route => {
    const index = (await route.request().postData())?.includes('retry-one.wav') ? 0 : 1;
    await route.fulfill({ json: project(ids[index], titles[index]) });
  });
  await page.route('**/api/batch/analyze', async route => route.fulfill({ json: {
    id: 'retry-fixture', status: 'queued', batch: true, progress: 0,
    items: ids.map((id, index) => ({ project: id, title: titles[index], status: 'queued', progress: 0 })),
  } }));
  await page.route('**/api/jobs/retry-fixture', async route => {
    pollCount += 1;
    if (pollCount === 1) return route.fulfill({ status: 503, json: { error: 'temporary fixture outage' } });
    return route.fulfill({ json: {
      id: 'retry-fixture', status: 'done', batch: true, progress: 100,
      message: '批量分析完成',
      items: ids.map((id, index) => ({ project: id, title: titles[index], status: 'done', progress: 100, message: '完成' })),
    } });
  });

  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles([
    audioFile('retry-one.wav'), audioFile('retry-two.wav'),
  ]);
  await page.locator('#batchAnalyze').click();

  await expect(page.locator('#message')).toContainText('正在重试');
  await expect(page.locator('#batchStatus')).toContainText('Retry Song Two');
  await expect(page.locator('#batchStatus')).toContainText('批次完成：2/2 首成功');
  expect(pollCount).toBe(2);
});

test('a missing batch job stops polling and releases the interface', async ({ page }) => {
  await page.route('**/api/import', async (route, request) => {
    const index = (await request.postData())?.includes('missing-one.wav') ? 0 : 1;
    await route.fulfill({ json: project(`missing-song-${index + 1}`, `Missing Song ${index + 1}`) });
  });
  await page.route('**/api/batch/analyze', async route => route.fulfill({ json: {
    id: 'missing-fixture', status: 'queued', batch: true, progress: 0,
    items: [1, 2].map(index => ({ project: `missing-song-${index}`, title: `Missing Song ${index}`, status: 'queued', progress: 0 })),
  } }));
  let pollCount = 0;
  await page.route('**/api/jobs/missing-fixture', async route => {
    pollCount += 1;
    await route.fulfill({ status: 404, json: { error: 'unknown job' } });
  });

  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles([
    audioFile('missing-one.wav'), audioFile('missing-two.wav'),
  ]);
  await page.locator('#batchAnalyze').click();

  await expect(page.locator('#message')).toContainText('HTTP 404');
  await expect(page.locator('#job')).toHaveClass(/hidden/);
  await expect(page.locator('#batchStatus')).toContainText('Missing Song 2');
  expect(pollCount).toBe(1);
});
