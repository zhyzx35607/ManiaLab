const { test, expect } = require('@playwright/test');

for (const action of ['expression', 'structure', 'recommendation']) {
  test(`${action} restores the draft clock without borrowing saved candidate diagnostics`, async ({ page }) => {
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto('/');
    await expect(page.locator('#recent option')).toContainText([`UI draft ${action}`]);
    await page.locator('#recent').selectOption({label:`UI draft ${action}`});
    await page.locator('#timingConfirmed').check();
    await page.locator('#generate').click();
    await expect(page.locator('#message')).toContainText('结构检查通过');
    await expect(page.locator('#timingReport')).toContainText('检查 1 段');

    async function apply() {
      if(action==='expression') {
        await page.locator('#expressionAuto').click();
        await expect(page.locator('#message')).toContainText('已保存逐段声部方案');
      } else if(action==='structure') {
        await page.locator('#structureBoundaries').fill('6');
        await page.locator('#sectionSave').click();
        await expect(page.locator('#message')).toContainText('已划分 2 个音乐段落');
      } else {
        await page.locator('#recommendationPanel').evaluate(el=>{el.open=true;});
        const response=page.waitForResponse('**/recommendations');
        await page.locator('#recommendationRefresh').click();
        await response;
        await page.locator('[data-recommendation="hybrid"]').click();
        await expect(page.locator('#message')).toContainText('已采用「');
      }
      await expect(page.locator('#generate')).toBeEnabled();
    }

    // A matching saved clock keeps its diagnostics and explicit confirmation.
    await apply();
    await expect(page.locator('#timingReport')).toContainText('检查 1 段');
    await expect(page.locator('#timingConfirmed')).toBeChecked();
    await page.locator('#bpm').fill('137');
    await page.locator('#bpm').blur();
    await page.locator('#audioShift').fill('50');
    await page.locator('#audioShift').blur();
    await page.locator('.audition-panel').evaluate(el=>{el.open=true;});
    await page.locator('#auditionOffset').fill('27');
    await page.locator('#timingConfirmed').check();
    await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
    await apply();
    await expect(page.locator('#bpm')).toHaveValue('137');
    await expect(page.locator('#audioShift')).toHaveValue('50');
    await expect(page.locator('#offset')).toHaveValue('-49');
    await expect(page.locator('#auditionOffset')).toHaveValue('27');
    await expect(page.locator('#timingConfirmed')).toBeChecked();
    await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
    await expect(page.locator('#timingReport')).not.toContainText('检查 1 段');
    expect(await page.evaluate(()=>timingSections)).toEqual([]);
    // Dependent suggestions must also be built for the restored BPM, not the
    // temporary saved candidate clock loaded during the response.
    await page.locator('#recommendationPanel').evaluate(el=>{el.open=true;});
    await expect(page.locator('[data-recommendation="hybrid"]')).toBeVisible();
    const applied=page.waitForResponse('**/apply-recommendation');
    await page.locator('[data-recommendation="hybrid"]').click();
    expect((await applied).ok()).toBeTruthy();
    await expect(page.locator('#generate')).toBeEnabled();
    await expect(page.locator('#bpm')).toHaveValue('137');
    await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
    // The retained fractional anchor must survive this restoration as well.
    await page.locator('#audioShift').fill('0');
    await page.locator('#audioShift').blur();
    await expect(page.locator('#offset')).toHaveValue('1');
    expect(errors).toEqual([]);
  });
}
