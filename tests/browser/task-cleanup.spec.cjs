const { test, expect } = require('@playwright/test');

test('a committed result keeps a visible cleanup warning and later success clears it', async ({ page }) => {
  await page.goto('/');
  const chart = `osu file format v14

[General]
Mode: 3

[Metadata]
Title: UI cleanup warning
Artist: Synthetic
Version: Fixture

[Difficulty]
CircleSize:6

[TimingPoints]
0,500,4,2,0,70,1,0

[HitObjects]
40,192,1000,128,0,2000:0:0:0:0:
450,192,3000,1,0,0:0:0:0:
`;
  await page.locator('#files').setInputFiles([{
    name: 'cleanup-warning.osu', mimeType: 'text/plain', buffer: Buffer.from(chart),
  }]);
  await expect(page.locator('#projectTitle')).toHaveText('UI cleanup warning');
  const warning = '结果已保存，但资源清理失败，请查看本地日志并重新打开项目核对';
  await page.route('**/api/jobs/*', async route => {
    const response = await route.fetch(), job = await response.json();
    if (job.status === 'done') Object.assign(job, { cleanup_failed: true, message: warning });
    await route.fulfill({ response, json: job });
  });
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText(warning);
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await expect(page.locator('#message')).toHaveClass(/error/);
  await expect(page.locator('#job')).toHaveClass(/hidden/);
  const candidateId = await page.evaluate(() => variant()?.candidate_id);
  expect(candidateId).toMatch(/^[a-f0-9]{32}$/);
  const exportUrl = new URL(await page.locator('#exportOsu').getAttribute('href'), page.url());
  expect(exportUrl.pathname).toMatch(/\/export\/osu$/);
  expect(exportUrl.searchParams.get('candidate_id')).toBe(candidateId);
  const exported = await page.request.get(exportUrl.href);
  expect(exported.ok()).toBeTruthy();
  expect(await exported.text()).toContain('1000,128,0,2000:');

  await page.unroute('**/api/jobs/*');
  await page.locator('#generate').click();
  await expect(page.locator('#variants option')).toHaveCount(2);
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await expect(page.locator('#message')).not.toContainText(warning);
  await expect(page.locator('#message')).not.toHaveClass(/error/);
});
