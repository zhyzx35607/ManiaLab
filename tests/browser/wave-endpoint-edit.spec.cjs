const { test, expect } = require('@playwright/test');

async function generateClock(page, shift) {
  const previous = await page.evaluate(() => variant()?.created);
  await page.locator('#audioShift').fill(String(shift));
  await page.locator('#audioShift').blur();
  await page.locator('#offset').fill(shift === 22.5 ? '93' : '190');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect.poll(() => page.evaluate(previous =>
    !busy && !!variant() && variant().created !== previous, previous)).toBe(true);
  await expect(page.locator('#message')).toContainText('结构检查通过');
  return page.evaluate(() => structuredClone(variant()));
}

async function selectNote(page, note) {
  await expect.poll(async () => {
    const box = await page.evaluate(note => {
      seek(Math.max(0, note.start / 1000 - .1));
      draw();
      const hit = hitboxes.find(item => item.n.id === note.id);
      return hit ? { x: hit.x + hit.w / 2, y: hit.y + hit.h / 2 } : null;
    }, note);
    if (!box) return false;
    await page.locator('#roll').click({ position: box });
    return page.evaluate(id => selected === id, note.id);
  }).toBe(true);
}

async function selectCandidate(page, candidate) {
  const index = await page.evaluate(created =>
    state.variants.findIndex(item => item.created === created), candidate.created);
  expect(index).toBeGreaterThanOrEqual(0);
  await page.locator('#variants').selectOption(String(index));
  await expect.poll(() => page.evaluate(() => variant().created)).toBe(candidate.created);
  await expect(page.locator('#audioShift')).toHaveValue(String(candidate.audio_shift_ms));
}

for (const clock of [
  { title: 'UI timing wave', shift: 22.5, start: 1218, end: 1718 },
  { title: 'UI timing clock', shift: -75, start: 1315, end: 1815 },
]) {
  test(`waveform endpoint buttons save an LN on the ${clock.shift} ms candidate clock`, async ({ page }) => {
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('/');
    await page.locator('#recent').selectOption({ label: clock.title });
    await expect(page.locator('#projectTitle')).toHaveText(clock.title);
    const other = await generateClock(page, clock.shift === 22.5 ? -75 : 22.5);
    const parent = await generateClock(page, clock.shift);
    const original = parent.notes.find(note => note.id === 'timing0');
    expect(original.end).toBeGreaterThan(original.start);
    await selectNote(page, original);
    await page.locator('#noteEditor').evaluate(element => { element.open = true; });
    await page.locator('#waveDetailPanel').evaluate(element => { element.open = true; });

    // The fixture's first raw interval is 1115–1865 ms. Shorten it at
    // physical audio positions 1240 and 1740 ms without introducing overlap.
    await page.locator('#waveClock').uncheck();
    await page.locator('#waveNote').click();
    await expect(page.locator('#waveCursor')).toHaveValue(String(original.start + clock.shift));
    await page.locator('#waveCursor').fill('1240');
    await page.locator('#editWaveStart').click();
    await expect(page.locator('#editStart')).toHaveValue(String(clock.start));

    await page.locator('#waveClock').check();
    await expect.poll(() => page.evaluate(() => waveDetail?.shift)).toBe(clock.shift);
    await page.locator('#waveCursor').fill(String(1740 - clock.shift));
    await page.locator('#editWaveEnd').click();
    await expect(page.locator('#editEnd')).toHaveValue(String(clock.end));
    await page.locator('#editSave').click();
    await expect.poll(() => page.evaluate(created =>
      !busy && variant().created !== created, parent.created)).toBe(true);
    await expect(page.locator('#message')).toContainText('手动修改已保存为新候选');
    const edited = await page.evaluate(() => structuredClone(variant()));
    const changed = edited.notes.find(note => note.id === original.id);
    expect(changed).toMatchObject({ start: clock.start, end: clock.end, lane: original.lane });
    expect(edited.source_evidence).toEqual(parent.source_evidence);
    expect(edited.edits.at(-1)).toMatchObject({
      parent_created: parent.created, before: original, after: changed,
    });

    // Switching through a candidate with the opposite clock must restore
    // that candidate's shift before locating its waveform endpoint.
    await selectCandidate(page, other);
    const otherNote = other.notes.find(note => note.id === original.id);
    await selectNote(page, otherNote);
    await page.locator('#waveClock').uncheck();
    await page.locator('#waveNote').click();
    await expect(page.locator('#waveCursor')).toHaveValue(String(otherNote.start + other.audio_shift_ms));
    await selectCandidate(page, parent);
    expect(await page.evaluate(() => variant().notes)).toEqual(parent.notes);
    await selectCandidate(page, edited);
    await selectNote(page, changed);
    await page.locator('#waveTail').click();
    await expect(page.locator('#waveCursor')).toHaveValue(String(clock.end + clock.shift));

    const pid = await page.locator('#recent').inputValue();
    const exported = await (await page.request.get(`/api/projects/${pid}/export/json`)).json();
    expect(exported.variant.notes.find(note => note.id === original.id)).toEqual(changed);
    await page.reload();
    await page.locator('#recent').selectOption(pid);
    await expect.poll(() => page.evaluate(() => variant()?.created)).toBe(edited.created);
    await expect(page.locator('#audioShift')).toHaveValue(String(clock.shift));
    expect(await page.evaluate(() => variant().notes.find(note => note.id === 'timing0'))).toEqual(changed);
    expect(errors).toEqual([]);
  });
}
