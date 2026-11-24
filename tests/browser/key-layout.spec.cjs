const { test, expect } = require('@playwright/test');

test('changing an 8K hand layout invalidates the old candidate until regeneration', async ({ page }) => {
  await page.goto('/');
  const chart = `osu file format v14\n\n[General]\nMode: 3\n\n[Metadata]\nTitle: Key layout\nArtist: Fixture\nVersion: Eight key\n\n[Difficulty]\nCircleSize: 8\n\n[TimingPoints]\n0,500,4,2,0,70,1,0\n\n[HitObjects]\n25,192,1003,128,0,1708:0:0:0:0:\n`;
  await page.locator('#files').setInputFiles([
    { name: 'layout.osu', mimeType: 'text/plain', buffer: Buffer.from(chart) },
  ]);
  await page.locator('#generate').click();
  await expect(page.locator('#variants option:checked')).toContainText('8K');
  await expect(page.locator('#exportOsu')).toHaveAttribute('href', /export\/osu\?candidate_id=/);

  await page.locator('#keyLayout').selectOption('8k_4_4');
  await expect(page.locator('#gridReport')).toContainText('目标键型 8k_4_4');
  await expect(page.locator('#exportOsu')).not.toHaveAttribute('href');

  await page.locator('#generate').click();
  await expect(page.locator('#exportOsu')).toHaveAttribute('href', /export\/osu\?candidate_id=/);
  const pid = await page.locator('#recent').inputValue();
  const project = await (await page.request.get(`/api/projects/${pid}`)).json();
  expect(project.variants[project.active].settings.key_layout).toBe('8k_4_4');
});
