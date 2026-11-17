const { test, expect } = require('@playwright/test');

async function generateEvidence(page, title) {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([title]);
  await page.locator('#recent').selectOption({ label:title });
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await page.locator('#reviewSummary').evaluate(el => { el.closest('details').open = true; });
}

async function selectNote(page, id, time) {
  await page.evaluate(time => seek(time), time);
  const box = await page.evaluate(id => {
    draw();const b = hitboxes.find(b => b.n.id === id);
    return b ? {x:b.x+b.w/2,y:b.y+b.h/2} : null;
  }, id);
  expect(box).not.toBeNull();
  await page.locator('#roll').click({position:box});
}

test('head and tail review locations use candidate time and the corresponding stem', async ({ page }) => {
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await generateEvidence(page,'UI evidence review');
  await expect(page.locator('#reviewSummary')).toContainText('14 个起音');
  await expect(page.locator('#reviewSummary')).toContainText('14 个 LN 尾端');
  await expect(page.locator('[data-review-kind="head"]')).toHaveCount(12);
  await expect(page.locator('[data-review-kind="tail"]')).toHaveCount(12);
  await page.locator('#voice').selectOption('piano');
  const first = page.locator('[data-review-kind="tail"]').first();
  const note = await page.evaluate(() => variant().notes.find(n=>n.id==='evidence0'));
  await first.click();
  await expect(page.locator('#voice')).toHaveValue('other');
  await expect(page.locator('#waveCursor')).toHaveValue(String(note.end));
  await expect(page.locator('#waveDetailInfo')).toContainText('other');
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）1710 ms');
  await expect(page.locator('#inspector')).toContainText('量化前表达尾（制谱时间）1688 ms');
  await expect(page.locator('#inspector')).toContainText('模型预测音尾');
  await expect(page.locator('#inspector')).not.toContainText('检测到释放');
  await expect.poll(()=>page.evaluate(()=>currentTime())).toBeCloseTo((note.end+22.5)/1000-.3,2);
  await page.locator('#waveClock').uncheck();
  await first.click();
  await expect(page.locator('#waveCursor')).toHaveValue(String(note.end+22.5));
  await page.locator('#audioShift').fill('0');await page.locator('#audioShift').blur();
  await page.locator('#waveClock').check();
  await first.click();
  await expect(page.locator('#waveCursor')).toHaveValue(String(note.end+22.5));
  expect(errors).toEqual([]);
});

test('manual tap extensions and additions stay distinct from detected endpoints', async ({ page }) => {
  await generateEvidence(page,'UI evidence edits');
  await selectNote(page,'unobserved',8.8);
  await page.locator('#noteEditor').evaluate(el=>{el.open=true;});
  await page.locator('#editEnd').fill('9478');
  await page.locator('#editSave').click();
  await expect(page.locator('#inspector')).toContainText('人工修改');
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）9600 ms');
  await expect(page.locator('#inspector')).toContainText('量化前表达尾（制谱时间）8978 ms');
  await expect(page.locator('#inspector')).toContainText('未观测到音尾');
  await page.locator('#editStart').fill('11040');
  await page.locator('#editEnd').fill('11540');
  await page.locator('#editLane').fill('1');
  await page.locator('#editAdd').click();
  await expect(page.locator('#inspector')).toContainText('人工添加');
  await expect(page.locator('#inspector')).not.toContainText('检测音高');
  await expect(page.locator('#inspector')).not.toContainText('频谱瞬态');
  await expect(page.locator('#inspector')).not.toContainText('量化前表达尾');
  await expect(page.locator('#inspector')).toContainText('没有采音证据');
  await expect(page.locator('#reviewSummary')).toContainText('14 个起音');
  await expect(page.locator('#reviewSummary')).toContainText('15 个 LN 尾端');
});

test('saved evidence survives changed analysis and legacy gaps remain unknown', async ({ page }) => {
  await generateEvidence(page,'UI evidence snapshots');
  const pid=await page.locator('#recent').inputValue();
  let legacy=false,missing=false;
  await page.route(`**/api/projects/${pid}`,async route=>{
    const response=await route.fetch(),data=await response.json();
    data.analysis.candidates.find(n=>n.id==='evidence0').end=7777;
    const v=data.variants[data.active];
    if(legacy)delete v.source_evidence;
    if(missing)delete v.selection.decisions;
    await route.fulfill({response,json:data});
  });
  async function reopen() {
    await page.reload();await page.locator('#recent').selectOption(pid);
    await selectNote(page,'evidence0',.8);
  }
  await reopen();
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）1710 ms');
  await expect(page.locator('#inspector')).not.toContainText('7777 ms');
  await page.locator('#evidenceView').click();
  await selectNote(page,'evidence0',.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）7777 ms');
  legacy=true;await reopen();
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）1710 ms');
  missing=true;await reopen();
  await expect(page.locator('#inspector')).toContainText('原始候选尾：未知');
  await expect(page.locator('#inspector')).not.toContainText('7777 ms');
  await selectNote(page,'unknown',9.8);
  await expect(page.locator('#inspector')).toContainText('音尾来源：未知');
});

test('regional rewrites never substitute fresh decisions for locked legacy evidence', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI evidence legacy region']);
  await page.locator('#recent').selectOption({label:'UI evidence legacy region'});
  await selectNote(page,'evidence0',.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）1710 ms');
  const before=await page.evaluate(()=>variant().notes.find(n=>n.id==='evidence0'));
  await page.locator('#regenerate').evaluate(el=>{el.closest('details').open=true;});
  await page.locator('#regionStart').fill('8.978');
  await page.locator('#regionEnd').fill('9.978');
  await page.locator('#timingConfirmed').check();
  await page.locator('#regenerate').click();
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  const current=await page.evaluate(()=>({note:variant().notes.find(n=>n.id==='evidence0'),
    snapshot:variant().source_evidence,region:variant().region,
    decision:variant().selection.proposal.decisions.find(n=>n.id==='evidence0')}));
  expect(current.note).toEqual(before);
  expect(current.region).toEqual([8978,9978]);
  expect(current.snapshot).not.toHaveProperty('evidence0');
  expect(current.decision).toMatchObject({decision:'selected',end:2310});
  await selectNote(page,'evidence0',.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾：未知');
  await expect(page.locator('#inspector')).not.toContainText('2310 ms');
  const pid=await page.locator('#recent').inputValue();
  // Read the real regional result as an older version that never saved snapshots.
  await page.route(`**/api/projects/${pid}`,async route=>{
    const response=await route.fetch(),data=await response.json();
    const v=data.variants[data.active];
    v.selection=v.selection.proposal;delete v.selection.scope;delete v.source_evidence;
    await route.fulfill({response,json:data});
  });
  await page.reload();await page.locator('#recent').selectOption(pid);
  await selectNote(page,'evidence0',.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾：未知');
  await expect(page.locator('#inspector')).not.toContainText('2310 ms');
  // generation.py uses chart starts in [region[0], region[1]): lower is fresh,
  // upper is locked, even though both have selected decisions in the new trace.
  const boundaries=await page.evaluate(()=>['unobserved','unknown'].map(id=>({
    start:variant().notes.find(n=>n.id===id).start,
    decision:variant().selection.decisions.find(n=>n.id===id).decision})));
  expect(boundaries).toEqual([{start:8978,decision:'selected'},{start:9978,decision:'selected'}]);
  await selectNote(page,'unobserved',8.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾（原音频）9600 ms');
  await selectNote(page,'unknown',9.8);
  await expect(page.locator('#inspector')).toContainText('原始候选尾：未知');
  await expect(page.locator('#inspector')).not.toContainText('10250 ms');
});
