const {test, expect} = require('@playwright/test');
const fs = require('node:fs/promises');

async function downloadAnalysis(page) {
  const pending = page.waitForEvent('download');
  await page.locator('#exportJson').click();
  const download = await pending;
  expect(download.suggestedFilename()).toBe('ManiaLab-analysis.json');
  return JSON.parse(await fs.readFile(await download.path(), 'utf8'));
}

test('manual section expression follows the saved candidate in the controls and JSON download', async ({page}) => {
  const recent = page.locator('#recent');
  const variants = page.locator('#variants');
  const rows = page.locator('#structureRows > details');
  const save = page.locator('#expressionSave');
  const generate = page.locator('#generate');
  const mode = index => rows.nth(index).locator('[data-expression-voice="vocals"]');
  const ratio = index => rows.nth(index).locator('[data-section-phrase-ln]');

  async function openSections() {
    await expect(rows).toHaveCount(2);
    for (const row of await rows.all()) {
      await row.locator('summary').click();
    }
  }

  async function saveExpression() {
    await save.click();
    await expect(page.locator('#message')).toContainText('已保存逐段声部方案');
    await expect(generate).toBeEnabled();
  }

  async function generateCandidate(count) {
    await generate.click();
    await expect(variants.locator('option')).toHaveCount(count);
    await expect(variants).toHaveValue(String(count - 1));
    await expect(variants).toBeEnabled();
    await expect(page.locator('#message')).toContainText('结构检查通过');
  }

  function manualSections(earlyMode, earlyRatio, lateMode, lateRatio) {
    return {
      source: 'manual',
      sections: [
        {start_ms: 0, end_ms: 6000, voices: {vocals: {mode: earlyMode}}, phrase_ln: earlyRatio},
        {start_ms: 6000, end_ms: 12000, voices: {vocals: {mode: lateMode}},
         ...(lateRatio === undefined ? {} : {phrase_ln: lateRatio})},
      ],
    };
  }

  async function switchCandidate(index) {
    await variants.selectOption(String(index));
    await expect(variants).toBeEnabled();
    await openSections();
  }

  function longNoteIds(analysis) {
    return analysis.variant.notes.filter(note => note.end > note.start).map(note => note.id);
  }

  await page.goto('/');
  await recent.selectOption({label: 'UI manual expression'});
  await openSections();

  await expect(ratio(0)).toBeDisabled();
  await mode(0).selectOption('phrase');
  await expect(ratio(0)).toBeEnabled();
  await ratio(0).press('End');
  await mode(1).selectOption('onsets');
  await expect(ratio(1)).toBeDisabled();
  await saveExpression();
  await generateCandidate(1);
  const first = await downloadAnalysis(page);
  expect(first.expression).toMatchObject(manualSections('phrase', 1, 'onsets'));
  expect(first.variant.selection.expression).toEqual(first.expression);
  expect(longNoteIds(first)).toEqual(['early-head']);

  await openSections();
  await ratio(0).press('Home');
  await mode(1).selectOption('phrase');
  await ratio(1).press('End');
  await saveExpression();
  const saved = await downloadAnalysis(page);
  expect(saved.expression).toMatchObject(manualSections('phrase', 0, 'phrase', 1));
  expect(saved.variant.candidate_id).toBe(first.variant.candidate_id);
  expect(saved.variant.selection.expression).toEqual(first.expression);
  expect(longNoteIds(saved)).toEqual(['early-head']);

  await generateCandidate(2);
  const second = await downloadAnalysis(page);
  expect(second.variant.candidate_id).not.toBe(first.variant.candidate_id);
  expect(second.expression).toEqual(saved.expression);
  expect(second.variant.selection.expression).toEqual(saved.expression);
  expect(longNoteIds(second)).toEqual(['late-head']);

  await switchCandidate(0);
  await expect(mode(0)).toHaveValue('phrase');
  await expect(ratio(0)).toHaveValue('1');
  await expect(mode(1)).toHaveValue('onsets');
  await expect(ratio(1)).toBeDisabled();
  const restoredFirst = await downloadAnalysis(page);
  expect(restoredFirst.variant.candidate_id).toBe(first.variant.candidate_id);
  expect(restoredFirst.expression).toEqual(first.expression);
  expect(restoredFirst.variant.selection.expression).toEqual(first.expression);
  expect(longNoteIds(restoredFirst)).toEqual(['early-head']);

  await switchCandidate(1);
  await expect(mode(0)).toHaveValue('phrase');
  await expect(ratio(0)).toHaveValue('0');
  await expect(mode(1)).toHaveValue('phrase');
  await expect(ratio(1)).toHaveValue('1');
  const restoredSecond = await downloadAnalysis(page);
  expect(restoredSecond.variant.candidate_id).toBe(second.variant.candidate_id);
  expect(restoredSecond.expression).toEqual(second.expression);
  expect(restoredSecond.variant.selection.expression).toEqual(second.expression);
  expect(longNoteIds(restoredSecond)).toEqual(['late-head']);
});

test('an outdated expression snapshot preserves history and can be replaced by a new draft', async ({page}) => {
  const rows = page.locator('#structureRows > details');
  const save = page.locator('#expressionSave');
  const automatic = page.locator('#expressionAuto');
  const variants = page.locator('#variants');
  const generate = page.locator('#generate');
  const boundaries = page.locator('#structureBoundaries');

  await page.goto('/');
  await page.locator('#recent').selectOption({label: 'UI stale expression'});
  await expect(rows).toHaveCount(2);
  await rows.first().locator('summary').click();
  await rows.first().locator('[data-expression-voice="vocals"]').selectOption('phrase');
  await rows.first().locator('[data-section-phrase-ln]').press('End');
  await save.click();
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(variants).toHaveValue('0');
  await expect(variants).toBeEnabled();
  const original = await downloadAnalysis(page);
  const history = original.variant.selection.expression;
  expect(history).toMatchObject({source: 'manual', sections: [{end_ms: 6000}, {start_ms: 6000}]});

  await boundaries.fill('5');
  await page.locator('#sectionSave').click();
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(variants).toHaveValue('1');
  await expect(variants).toBeEnabled();
  await variants.selectOption('0');
  await expect(variants).toBeEnabled();
  await expect(boundaries).toHaveValue('5');
  await expect(rows).toHaveCount(0);
  await expect(save).toBeDisabled();
  await expect(automatic).toBeEnabled();
  const outdated = await downloadAnalysis(page);
  expect(outdated.expression).toBeNull();
  expect(outdated.variant.candidate_id).toBe(original.variant.candidate_id);
  expect(outdated.variant.selection.expression).toEqual(history);

  await automatic.click();
  await expect(rows).toHaveCount(2);
  await expect(save).toBeEnabled();
  const recovered = await downloadAnalysis(page);
  expect(recovered.expression).toMatchObject({
    source: 'automatic', sections: [{end_ms: 5000}, {start_ms: 5000}],
  });
  expect(recovered.variant.candidate_id).toBe(original.variant.candidate_id);
  expect(recovered.variant.selection.expression).toEqual(history);
});
