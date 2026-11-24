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

test('batch analysis imports three songs, caps parallelism, and isolates a failed song', async ({ page }) => {
  const imported = [];
  let batchRequest;
  let pollCount = 0;
  const ids = ['batch-song-1', 'batch-song-2', 'batch-song-3'];
  const titles = ['Song One', 'Song Two', 'Song Three'];

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
    const items = [
      { project: ids[0], title: titles[0], status: 'done', progress: 100, message: '完成', result: { candidates: 12 } },
      { project: ids[1], title: titles[1], status: 'error', progress: 35, message: '合成分析失败' },
      {
        project: ids[2], title: titles[2], status: finished ? 'done' : 'running',
        progress: finished ? 100 : 60, message: finished ? '完成' : '正在提取音符',
        ...(finished ? { result: { candidates: 9 } } : {}),
      },
    ];
    await route.fulfill({ json: {
      id: 'batch-fixture', status: finished ? 'done' : 'running', batch: true,
      progress: finished ? 100 : 65, message: finished ? '批量分析完成' : '正在分析 2 / 3', items,
    } });
  });
  await page.route(/\/api\/projects\/batch-song-1$/, async route => {
    await route.fulfill({ json: project(ids[0], titles[0]) });
  });
  await page.route('**/api/projects/batch-song-1/audio*', route => route.fulfill({ status: 404, body: '' }));

  await page.goto('/');
  await page.locator('.batch-panel > summary').click();
  await page.locator('#batchFiles').setInputFiles([
    audioFile('one.wav'), audioFile('two.wav'), audioFile('three.wav'),
  ]);
  await page.locator('#batchAnalyze').click();

  await expect(page.locator('#batchStatus [data-batch-item]')).toHaveCount(3);
  await expect(page.locator('#batchStatus')).toContainText('Song One');
  await expect(page.locator('#batchStatus')).toContainText('Song Two');
  await expect(page.locator('#batchStatus')).toContainText('Song Three');
  await expect(page.locator('#batchStatus')).toContainText('完成');
  await expect(page.locator('#batchStatus')).toContainText('失败');

  expect(imported).toHaveLength(3);
  expect(imported[0]).toContain('one.wav');
  expect(imported[1]).toContain('two.wav');
  expect(imported[2]).toContain('three.wav');
  expect(batchRequest.projects).toEqual(ids);
  expect(batchRequest.max_parallel).toBe(2);

  await page.locator('[data-open-project="batch-song-1"]').click();
  await expect(page.locator('#projectTitle')).toHaveText('Song One');
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
