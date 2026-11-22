const { test, expect } = require('@playwright/test');

test('a pending recent load cannot replace a project while its candidate is generating', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option').filter({ hasText: 'UI project race generation' })).toHaveCount(1);
  await expect(page.locator('#recent option').filter({ hasText: 'UI timing clock' })).toHaveCount(1);
  const projects = await page.locator('#recent option').evaluateAll(options =>
    options.map(option => ({ id: option.value, title: option.textContent.trim() })),
  );
  const projectA = projects.find(project => project.title === 'UI project race generation');
  const projectB = projects.find(project => project.title === 'UI timing clock');
  expect(projectA).toBeTruthy();
  expect(projectB).toBeTruthy();

  await page.locator('#recent').selectOption(projectA.id);
  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);

  let releaseRecent;
  let markRecentRequested;
  const recentRequested = new Promise(resolve => { markRecentRequested = resolve; });
  const recentGate = new Promise(resolve => { releaseRecent = resolve; });
  let releaseJobPoll;
  let markJobPollRequested;
  const jobPollRequested = new Promise(resolve => { markJobPollRequested = resolve; });
  const jobPollGate = new Promise(resolve => { releaseJobPoll = resolve; });
  let heldJobPoll = false;

  await page.route(`**/api/projects/${projectB.id}`, async route => {
    markRecentRequested();
    await recentGate;
    await route.continue();
  });
  await page.route('**/api/jobs/**', async route => {
    if (!heldJobPoll && route.request().method() === 'GET') {
      heldJobPoll = true;
      markJobPollRequested();
      await jobPollGate;
    }
    await route.continue();
  });

  await page.locator('#recent').selectOption(projectB.id);
  await recentRequested;
  const generateStarted = page.waitForRequest(request =>
    request.url().endsWith(`/api/projects/${projectA.id}/generate`) && request.method() === 'POST',
  );
  await page.locator('#generate').click();
  await generateStarted;
  await jobPollRequested;

  releaseRecent();
  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);
  releaseJobPoll();

  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);
  await expect(page.locator('#variants option')).toHaveCount(1);
});

test('a pending recent load cannot replace a project while its expression is saving', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option').filter({ hasText: 'UI project race expression' })).toHaveCount(1);
  await expect(page.locator('#recent option').filter({ hasText: 'UI timing clock' })).toHaveCount(1);
  const projects = await page.locator('#recent option').evaluateAll(options =>
    options.map(option => ({ id: option.value, title: option.textContent.trim() })),
  );
  const projectA = projects.find(project => project.title === 'UI project race expression');
  const projectB = projects.find(project => project.title === 'UI timing clock');
  expect(projectA).toBeTruthy();
  expect(projectB).toBeTruthy();

  await page.locator('#recent').selectOption(projectA.id);
  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);

  let releaseRecent;
  let markRecentRequested;
  const recentRequested = new Promise(resolve => { markRecentRequested = resolve; });
  const recentGate = new Promise(resolve => { releaseRecent = resolve; });
  let releaseSave;
  let markSaveRequested;
  const saveRequested = new Promise(resolve => { markSaveRequested = resolve; });
  const saveGate = new Promise(resolve => { releaseSave = resolve; });

  await page.route(`**/api/projects/${projectB.id}`, async route => {
    markRecentRequested();
    await recentGate;
    await route.continue();
  });
  await page.route(`**/api/projects/${projectA.id}/expression`, async route => {
    markSaveRequested();
    await saveGate;
    await route.continue();
  });

  await page.locator('#recent').selectOption(projectB.id);
  await recentRequested;
  await page.locator('#expressionAuto').click();
  await saveRequested;

  releaseRecent();
  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);
  releaseSave();

  await expect(page.locator('#projectTitle')).toHaveText(projectA.title);
  await expect(page.locator('#message')).toContainText('已保存逐段声部方案');
  await expect(page.locator('#expressionSave')).toBeEnabled();
});
