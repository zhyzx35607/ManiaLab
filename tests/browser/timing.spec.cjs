const { test, expect } = require('@playwright/test');

async function openTiming(page, title = 'UI timing clock') {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([title]);
  await page.locator('#recent').selectOption({ label: title });
  await expect(page.locator('#offset')).toHaveValue('93');
  await expect(page.locator('#audioShift')).toHaveValue('22.5');
}

test('a delayed timing result stays expired even after returning to the same clock', async ({ page }) => {
  await openTiming(page);
  let release;
  const blocked = new Promise(resolve => { release = resolve; });
  let requested;
  const started = new Promise(resolve => { requested = resolve; });
  await page.route('**/timing-check', async route => {
    const clock = route.request().postDataJSON();
    requested();
    await blocked;
    await route.fulfill({ json:{ sections:[{start_ms:0,end_ms:12000,attacks:20,
      suspect:false,phase_error_ms:0,coverage:1}], suspect:0,status:'available',
      insufficient_windows:[],clock,window_clock:'audio' } });
  });
  await page.locator('#checkTiming').click();
  await started;
  await page.locator('#bpm').fill('137');
  await page.locator('#bpm').fill('120');
  const completed = page.waitForResponse('**/timing-check');
  release();
  await completed;
  await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
  await expect(page.locator('#timingReport')).not.toContainText('0 段可能错位');
});

test('saved diagnostics return only for the candidate clock and legacy projects remain readable', async ({ page }) => {
  await openTiming(page, 'UI timing saved');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await expect(page.locator('#timingReport')).toContainText('检查 1 段');
  const pid = await page.locator('#recent').inputValue();
  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#timingReport')).toContainText('检查 1 段');
  await page.route(`**/api/projects/${pid}`, async route => {
    const response = await route.fetch(), data = await response.json();
    data.variants[data.active].timing_diagnostics.clock.bpm += 1;
    await route.fulfill({ response, json:data });
  });
  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
  await page.unroute(`**/api/projects/${pid}`);
  await page.route(`**/api/projects/${pid}`, async route => {
    const response = await route.fetch(), data = await response.json();
    delete data.variants[data.active].timing_diagnostics;
    await route.fulfill({ response, json:data });
  });
  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#timingReport')).toContainText('旧候选未记录完整诊断');
});

test('precision BPM candidate preserves its adjusted clock through confirmation, generation, and reload', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI timing precision']);
  await page.locator('#recent').selectOption({ label: 'UI timing precision' });

  const preciseOption = page.locator('#tempoCandidates option').filter({ hasText: '120.37 BPM' });
  await expect(preciseOption).toHaveCount(1);
  await page.locator('#tempoCandidates').selectOption(await preciseOption.getAttribute('value'));
  await expect(page.locator('#bpm')).toHaveValue('120.37');
  await expect(page.locator('#audioShift')).toHaveValue('22.5');
  await expect(page.locator('#offset')).toHaveValue('93');
  await expect(page.locator('#timingConfirmed')).not.toBeChecked();

  await page.locator('#audioShift').fill('37.5');
  await page.locator('#audioShift').blur();
  await expect(page.locator('#offset')).toHaveValue('78');
  await expect(page.locator('#timingConfirmed')).not.toBeChecked();
  const pid = await page.locator('#recent').inputValue();

  const rejectedRequest = page.waitForRequest(request =>
    request.url().endsWith(`/api/projects/${pid}/generate`) && request.method() === 'POST');
  await page.locator('#generate').click();
  expect((await rejectedRequest).postDataJSON()).toMatchObject({
    bpm: 120.37, offset: 78, audio_shift_ms: 37.5, timing_confirmed: false,
  });
  await expect(page.locator('#message')).toContainText('当前自动 Timing 不可靠');
  let saved = await (await page.request.get(`/api/projects/${pid}`)).json();
  expect(saved.variants).toHaveLength(0);

  await page.locator('#timingConfirmed').check();
  const acceptedRequest = page.waitForRequest(request =>
    request.url().endsWith(`/api/projects/${pid}/generate`) && request.method() === 'POST');
  await page.locator('#generate').click();
  expect((await acceptedRequest).postDataJSON()).toMatchObject({
    bpm: 120.37, offset: 78, audio_shift_ms: 37.5, timing_confirmed: true,
  });
  await expect(page.locator('#message')).toContainText('结构检查通过');
  saved = await (await page.request.get(`/api/projects/${pid}`)).json();
  expect(saved.variants).toHaveLength(1);
  expect(saved.variants[0]).toMatchObject({
    bpm: 120.37, offset: 78, audio_shift_ms: 37.5, timing_confirmed: true,
  });

  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#bpm')).toHaveValue('120.37');
  await expect(page.locator('#offset')).toHaveValue('78');
  await expect(page.locator('#audioShift')).toHaveValue('37.5');
  await expect(page.locator('#timingConfirmed')).toBeChecked();
  expect(errors).toEqual([]);
});

test('automatic clock projection and manual offset input keep their rounding policies', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI timing offset rounding']);
  await page.locator('#recent').selectOption({ label: 'UI timing offset rounding' });
  await expect(page.locator('#offset')).toHaveValue('21');

  const candidate = page.locator('#tempoCandidates option').filter({ hasText: '120 BPM · 偏移 21 ms' });
  await expect(candidate).toHaveCount(1);
  await page.locator('#tempoCandidates').selectOption(await candidate.getAttribute('value'));
  await page.locator('#audioShift').fill('22.5');
  await page.locator('#audioShift').blur();

  await expect(page.locator('#offset')).toHaveValue('-1');
  await expect(page.locator('#tempoCandidates')).toContainText('偏移 -1 ms');
  expect(await page.evaluate(() => chartOffset(-1.5))).toBe(-2);

  await page.locator('#referenceTiming').fill('BPM: 120 Offset: -1.5');
  await page.locator('#applyTiming').click();
  await expect(page.locator('#offset')).toHaveValue('-2');
  await expect(page.locator('#message')).toContainText('Offset -2 ms');
});

for (const edit of ['bpm', 'division', 'halfBpm', 'doubleBpm']) {
  test(`${edit} preserves the original offset anchor before changing the lead`, async ({ page }) => {
    await openTiming(page);
    if (edit === 'bpm') await page.locator('#bpm').fill('121');
    else if (edit === 'division') await page.locator('#division').selectOption('8');
    else await page.locator(`#${edit}`).click();
    await page.locator('#audioShift').fill('0');
    await page.locator('#audioShift').blur();
    await expect(page.locator('#offset')).toHaveValue('115');
  });
}

test('manual fractional offsets are visibly normalized before checking the same clock', async ({ page }) => {
  await openTiming(page);
  for (const [value, expected] of [['93.5',94], ['94.5',94], ['-93.5',-94], ['-94.5',-94], ['93.2',93]]) {
    await page.locator('#offset').fill(value);
    const completed = page.waitForResponse('**/timing-check');
    await page.locator('#checkTiming').click();
    const response = await completed;
    expect((await response.json()).clock.offset).toBe(expected);
    await expect(page.locator('#offset')).toHaveValue(String(expected));
    await expect(page.locator('#timingReport')).toContainText('检查 1 段');
    await expect(page.locator('#message')).toContainText(`Offset 已取整为 ${expected} ms`);
  }
  await page.locator('#offset').fill('94.5');
  const refitted = page.waitForRequest('**/timing-refit');
  await page.locator('#refitTiming').click();
  expect((await refitted).postDataJSON().offset).toBe(94);
  await expect(page.locator('#timingReport')).toContainText('重新估计 Offset');
});

test('reference preferences preserve a draft clock without restoring candidate diagnostics', async ({ page }) => {
  await openTiming(page, 'UI timing style');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#timingReport')).toContainText('检查 1 段');
  await page.locator('#bpm').fill('137');
  const chart = `osu file format v14\n\n[General]\nAudioFilename: song.wav\nMode: 3\n\n[Metadata]\nTitle: Timing preference fixture\nArtist: Test\nVersion: Fixture\n\n[Difficulty]\nCircleSize:10\n\n[TimingPoints]\n0,500,4,2,0,70,1,0\n\n[HitObjects]\n`+
    Array.from({length:20},(_,i)=>`${25+i%10*51},192,${1000+i*500},128,0,${1300+i*500}:0:0:0:0:`).join('\n');
  await page.locator('#referenceStyleFile').setInputFiles([{name:'style.osu',mimeType:'text/plain',buffer:Buffer.from(chart)}]);
  await expect(page.locator('#message')).toContainText('已套用参考谱');
  await expect(page.locator('#bpm')).toHaveValue('137');
  await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
  await expect(page.locator('#timingReport')).not.toContainText('0 段可能错位');
});

test('timing results expire with the clock and report insufficient evidence explicitly', async ({ page }) => {
  await openTiming(page);
  let status = 'available';
  await page.route('**/timing-check', async route => {
    const clock = route.request().postDataJSON();
    const sections = status === 'insufficient' ? [] : [{ start_ms:0, end_ms:12000, attacks:20,
      suspect:false, phase_error_ms:0, coverage:1, evidence_source:'mix', fallback:true }];
    await route.fulfill({ json: { sections, suspect:0, status, clock, window_clock:'audio',
      insufficient_windows:status === 'available' ? [] : [{ start_ms:12000, end_ms:24000, attacks:2 }] } });
  });
  await page.locator('#checkTiming').click();
  await expect(page.locator('#timingReport')).toContainText('检查 1 段');
  await page.locator('#bpm').fill('137');
  await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
  await expect(page.locator('#timingReport')).not.toContainText('0 段可能错位');
  await expect(page.locator('#tempoNote')).toContainText('原始估计');
  await expect(page.locator('#tempoNote')).toContainText('120 BPM');
  status = 'insufficient';
  await page.locator('#checkTiming').click();
  await expect(page.locator('#timingReport')).toContainText('证据不足，不能判断');
  await expect(page.locator('#timingReport')).not.toContainText('0 段可能错位');
  status = 'partial';
  await page.locator('#checkTiming').click();
  await expect(page.locator('#timingReport')).toContainText('1 个窗口证据不足');
});

test('raw and shifted wave views align the same physical point and locate candidate endpoints', async ({ page }) => {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await openTiming(page, 'UI timing wave');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  const note = await page.evaluate(() => variant().notes.find(n => n.id === 'timing0'));
  const box = await page.evaluate(() => {
    const b = hitboxes.find(b => b.n.id === 'timing0');
    return { x:b.x+b.w/2, y:b.y+b.h/2 };
  });
  await page.locator('#roll').click({ position:box });
  await page.locator('#waveDetailPanel').evaluate(element => { element.open = true; });
  await page.locator('#waveNote').click();
  await expect(page.locator('#waveCursor')).toHaveValue(String(note.start));
  await page.locator('#waveClock').uncheck();
  await page.locator('#waveNote').click();
  await expect.poll(async () => Number(await page.locator('#waveCursor').inputValue()))
    .toBeCloseTo(note.start+22.5, 0);
  const drawn = await page.evaluate(() => {
    const el = document.getElementById('waveDetail'), c = el.getContext('2d'), records = [];
    const move = c.moveTo;
    c.moveTo = function(x,y) { records.push({ color:this.strokeStyle, x,y }); return move.call(this,x,y); };
    try { drawWaveDetail(); } finally { c.moveTo = move; }
    const width = el.getBoundingClientRect().width-24;
    return records.map(r => ({ ...r, time:waveDetail.start_ms+(r.x-12)/width*(waveDetail.end_ms-waveDetail.start_ms) }));
  });
  expect(drawn.find(r => r.color === '#eef4ff').time).toBeCloseTo(note.start+22.5, 3);
  expect(drawn.find(r => r.color === '#78e4c4').time).toBeCloseTo(note.end+22.5, 3);
  expect(drawn.some(r => r.color === '#6b8c82' && Math.abs(r.time-note.start-22.5)<.001)).toBeTruthy();
  await page.locator('#waveTail').click();
  await expect.poll(async () => Number(await page.locator('#waveCursor').inputValue()))
    .toBeCloseTo(note.end+22.5, 0);
  await page.locator('#waveCursor').fill('1115');
  await expect(page.locator('#waveDetailInfo')).toContainText('距最近细分线 -0.50 ms');
  await page.locator('#waveAlign').click();
  await expect(page.locator('#offset')).toHaveValue('92');
  await page.locator('#waveClock').check();
  await page.locator('#waveCursor').fill('1092.5');
  await page.locator('#waveAlign').click();
  await expect(page.locator('#offset')).toHaveValue('92');
  expect(errors).toEqual([]);
});
