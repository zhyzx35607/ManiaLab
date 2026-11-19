const { test, expect } = require('@playwright/test');

test('audio candidate: generate, change keys, edit, reopen and export', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['打开已保存项目', 'UI synthetic']);
  await page.locator('#recent').selectOption({ label: 'UI synthetic' });
  await expect(page.locator('#projectTitle')).toHaveText('UI synthetic');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await expect(page.locator('#exportOsz')).toHaveAttribute('href', /export\/osz$/);
  await page.locator('[data-key="6"]').click();
  await expect(page.locator('#gridReport')).toContainText('当前预览仍是 10K');
  await expect(page.locator('#exportOsz')).not.toHaveAttribute('href');
  await page.locator('#generate').click();
  await expect(page.locator('#variants option:checked')).toContainText('6K');
  await expect(page.locator('#exportOsz')).toHaveAttribute('href', /export\/osz$/);
  await page.locator('#noteEditor').evaluate(element => { element.open = true; });
  await page.locator('#editStart').fill('11500');
  await page.locator('#editEnd').fill('11500');
  await page.locator('#editLane').fill('1');
  await page.locator('#editAdd').click();
  await expect(page.locator('#variants option:checked')).toContainText('已修正 1 次');
  const pid = await page.locator('#recent').inputValue();
  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#variants option:checked')).toContainText('已修正 1 次');
  await page.locator('#bpm').fill('121');
  await expect(page.locator('#exportOsz')).not.toHaveAttribute('href');
  await page.locator('#bpm').fill('120');
  await expect(page.locator('#exportOsz')).toHaveAttribute('href', /export\/osz$/);
  const downloadPromise = page.waitForEvent('download');
  await page.locator('#exportOsz').click();
  const download = await downloadPromise;
  expect(await download.failure()).toBeNull();
  const exported = await page.request.get(`/api/projects/${pid}/export/osu`);
  expect(exported.ok()).toBeTruthy();
  expect(await exported.text()).toMatch(/CircleSize:\s*6/);
  expect(errors).toEqual([]);
});

test('imported chart retains strict times after changing target keys', async ({ page }) => {
  await page.goto('/');
  const chart = `osu file format v14\n\n[General]\nAudioFilename: song.wav\nMode: 3\n\n[Metadata]\nTitle: Browser import\nArtist: Test\nVersion: Fixture\n\n[Difficulty]\nCircleSize:10\n\n[TimingPoints]\n0,500,4,2,0,70,1,0\n\n[HitObjects]\n25,192,1003,128,0,2007:0:0:0:0:\n486,192,3009,1,0,0:0:0:0:\n`;
  await page.locator('#files').setInputFiles([{ name: 'chart.osu', mimeType: 'text/plain', buffer: Buffer.from(chart) }]);
  await expect(page.locator('#projectTitle')).toHaveText('Browser import');
  await page.locator('#generate').click();
  await expect(page.locator('#exportOsu')).toHaveAttribute('href', /export\/osu$/);
  await page.locator('[data-key="6"]').click();
  await expect(page.locator('#exportOsu')).not.toHaveAttribute('href');
  await page.locator('#generate').click();
  await expect(page.locator('#variants option:checked')).toContainText('6K');
  const response = await page.request.get(await page.locator('#exportOsu').getAttribute('href'));
  const content = await response.text();
  expect(content).toContain(',1003,128,0,2007:');
  expect(content).toContain(',3009,1,0,');
  expect(content).toMatch(/CircleSize:\s*6/);
});

test('switching to a chart with missing audio shows a recovery instruction', async ({ page }) => {
  await page.goto('/');
  const chart = audio => `osu file format v14\n\n[General]\nAudioFilename: ${audio}\nMode: 3\n\n[Metadata]\nTitle: Audio mapping\nArtist: Fixture\nVersion: Test\n\n[Difficulty]\nCircleSize:7\n\n[TimingPoints]\n0,500,4,2,0,70,1,0\n\n[HitObjects]\n25,192,1003,128,0,2007:0:0:0:0:\n`;
  await page.locator('#files').setInputFiles([
    { name: 'a.osu', mimeType: 'text/plain', buffer: Buffer.from(chart('song.wav')) },
    { name: 'b.osu', mimeType: 'text/plain', buffer: Buffer.from(chart('missing.wav')) },
    { name: 'song.wav', mimeType: 'audio/wav', buffer: Buffer.from('synthetic audio') },
  ]);
  await expect(page.locator('#projectTitle')).toHaveText('Audio mapping');
  await expect(page.locator('#audioStatus')).toBeHidden();
  await page.locator('#charts').selectOption('b.osu');
  await expect(page.locator('#audioStatus')).toBeVisible();
  await expect(page.locator('#audioStatus')).toContainText('不会自动借用其他难度的音轨');
  await expect(page.locator('#audioStatus')).toContainText('重新导入此 .osu 文件和对应音频');
  await expect(page.locator('#audio')).not.toHaveAttribute('src', /.+/);
});
