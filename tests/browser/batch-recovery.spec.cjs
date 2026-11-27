const { test, expect } = require('@playwright/test');

test('refreshing the workbench restores a persistent batch and resumes only unfinished songs', async ({ page }) => {
  const finished = {
    project: 'completed-song',
    title: 'Completed Song',
    status: 'done',
    progress: 100,
    candidate_id: 'a'.repeat(32),
  };
  const interrupted = {
    project: 'interrupted-song',
    title: 'Interrupted Song',
    status: 'interrupted',
    progress: 42,
    message: '上次运行中断，可续跑。',
  };
  const previous = {
    id: 'persistent-batch',
    batch: true,
    resume_available: true,
    resume_kind: 'v32-generation',
    status: 'interrupted',
    created_at: 1,
    items: [finished, interrupted],
  };
  let resumeRequest;
  let recentItems = [previous];
  await page.route('**/api/batch/jobs?limit=12', route => route.fulfill({ json: { items: recentItems } }));
  await page.route('**/api/batch/jobs/persistent-batch', route => route.fulfill({ json: previous }));
  await page.route('**/api/batch/jobs/persistent-batch/resume', async route => {
    resumeRequest = route.request().postDataJSON();
    recentItems = [];
    await route.fulfill({ json: {
      id: 'resumed-generation',
      batch: true,
      resume_kind: 'v32-generation',
      status: 'queued',
      progress: 0,
      generation_total: 1,
      items: [{ ...interrupted, status: 'queued', progress: 0 }],
    } });
  });
  await page.route('**/api/jobs/resumed-generation', route => route.fulfill({ json: {
    id: 'resumed-generation',
    batch: true,
    resume_kind: 'v32-generation',
    status: 'done',
    progress: 100,
    items: [{ ...interrupted, status: 'done', progress: 100, result: { candidate_id: 'b'.repeat(32) } }],
  } }));
  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  await expect(page.locator('#batchHistory')).toContainText('批量生成');
  await expect(page.locator('#batchHistory')).toContainText('已中断');
  await page.locator('[data-resume-batch="persistent-batch"]').click();

  await expect.poll(() => resumeRequest).toEqual({});
  await expect(page.locator('#batchStatus')).toContainText('统一生成结束：1/1 首成功');
  await expect(page.locator('#batchStatus')).toContainText('Completed Song');
  await expect(page.locator('#batchStatus')).toContainText('Interrupted Song');
  await expect(page.locator('#batchStatus [data-batch-download]')).toHaveCount(2);
  await expect(page.locator('#batchStatus [data-batch-download]').nth(0))
    .toHaveAttribute('href', new RegExp(`completed-song.*candidate_id=${'a'.repeat(32)}`));
  await expect(page.locator('#batchStatus [data-batch-download]').nth(1))
    .toHaveAttribute('href', new RegExp(`interrupted-song.*candidate_id=${'b'.repeat(32)}`));
});

test('page load reconnects an active V32 batch and restores finished song downloads', async ({ page }) => {
  const active = {
    id: 'active-batch',
    batch: true,
    resume_kind: 'v32-generation',
    status: 'running',
    progress: 55,
    generation_total: 2,
    items: [
      {
        project: 'active-completed',
        title: 'Active Completed Song',
        status: 'done',
        progress: 100,
        candidate_id: 'c'.repeat(32),
      },
      {
        project: 'active-running',
        title: 'Active Running Song',
        status: 'running',
        progress: 25,
      },
    ],
  };
  let pollCount = 0;
  let reconnectAvailable = true;
  await page.route('**/api/batch/jobs?limit=12', route => route.fulfill({
    json: { items: [{
      ...active,
      status: reconnectAvailable ? 'running' : 'done',
      reconnect_available: reconnectAvailable,
      resume_available: false,
    }] },
  }));
  await page.route('**/api/batch/jobs/active-batch', route => route.fulfill({ json: active }));
  await page.route('**/api/jobs/active-batch', async route => {
    pollCount += 1;
    if (pollCount === 1) {
      await route.fulfill({ json: active });
      return;
    }
    reconnectAvailable = false;
    await route.fulfill({ json: {
      ...active,
      status: 'done',
      progress: 100,
      items: [
        active.items[0],
        {
          project: 'active-running',
          title: 'Active Running Song',
          status: 'done',
          progress: 100,
          result: { candidate_id: 'd'.repeat(32) },
        },
      ],
    } });
  });

  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();

  await expect(page.locator('#batchStatus')).toContainText('统一生成结束：2/2 首成功');
  await expect(page.locator('#batchHistory')).toContainText('批量生成');
  await expect(page.locator('#batchStatus')).toContainText('Active Completed Song');
  await expect(page.locator('#batchStatus')).toContainText('Active Running Song');
  await expect(page.locator('#batchStatus [data-batch-download]')).toHaveCount(2);
  expect(pollCount).toBeGreaterThan(1);
});

test('terminal batch history restores lineage downloads without polling', async ({ page }) => {
  const completed = {
    project: 'history-completed-song',
    title: 'History Completed Song',
    status: 'done',
    progress: 100,
    candidate_id: 'e'.repeat(32),
  };
  const previous = {
    id: 'terminal-results-batch',
    batch: true,
    resume_kind: 'v32-generation',
    status: 'done',
    created_at: 1,
    resume_available: false,
    reconnect_available: false,
    view_results_available: true,
    items: [completed],
  };
  let detailRequests = 0;
  let pollRequests = 0;
  let bundleItems;
  await page.route('**/api/batch/jobs?limit=12', route => route.fulfill({
    json: { items: [previous] },
  }));
  await page.route('**/api/batch/jobs/terminal-results-batch', async route => {
    detailRequests += 1;
    await route.fulfill({ json: previous });
  });
  await page.route('**/api/jobs/**', async route => {
    pollRequests += 1;
    await route.fulfill({ json: { status: 'done', items: [] } });
  });
  await page.route('**/api/batch/export-osz?*', async route => {
    bundleItems = new URL(route.request().url()).searchParams.get('items');
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
  await page.locator('[data-show-batch-results="terminal-results-batch"]').click();

  await expect(page.locator('#batchStatus')).toContainText('History Completed Song');
  await expect(page.locator('#batchStatus [data-batch-download]')).toHaveCount(1);
  await expect(page.locator('#batchStatus [data-batch-download]'))
    .toHaveAttribute('href', new RegExp(`history-completed-song.*candidate_id=${'e'.repeat(32)}`));
  await expect(page.locator('#batchDownloadAll')).toBeVisible();
  await expect(page.locator('#batchGenerate')).toBeDisabled();
  expect(detailRequests).toBe(1);
  expect(pollRequests).toBe(0);

  const downloadPromise = page.waitForEvent('download');
  await page.locator('#batchDownloadAll').click();
  expect(await (await downloadPromise).failure()).toBeNull();
  expect(bundleItems).toBe(`history-completed-song:${'e'.repeat(32)}`);
  expect(pollRequests).toBe(0);
});
