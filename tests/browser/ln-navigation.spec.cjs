const { test, expect } = require('@playwright/test');

async function openLNEvents(page, mode) {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([`UI LN navigation ${mode}`]);
  await page.locator('#recent').selectOption({label:`UI LN navigation ${mode}`});
  if(mode==='audio')await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await page.locator('#lnEventFilter').selectOption('release_only_rows');
  await expect(page.locator('#lnDesignEvents button').first()).toBeVisible();
  return page.evaluate(()=>variant().report.ln_patterns.locations.find(x=>x.kind==='release_only_rows'));
}

async function assertLocation(page, event, displayTime, audioTime) {
  await page.locator('#lnDesignEvents button').first().click();
  await expect(page.locator('#waveCursor')).toHaveValue(String(displayTime));
  const wave=await page.evaluate(()=>({start:waveDetail.start_ms+waveDetail.shift,
    span:Number(document.getElementById('waveSpan').value),paused:audio.paused}));
  expect(wave.start).toBeCloseTo(Math.max(0,audioTime-wave.span*.25),1);
  expect(wave.paused).toBeTruthy();
  // Transport and audition stay on exported chart time, independently of the
  // original-audio location used for waveform inspection.
  await expect.poll(()=>page.evaluate(()=>currentTime())).toBeCloseTo(Math.max(0,event.start_ms/1000-.6),3);
  await expect(page.locator('#auditionStart')).toHaveValue(Math.max(0,event.start_ms/1000-1).toFixed(1));
}

async function assertAuditionStart(page, event) {
  await page.locator('.audition-panel').evaluate(el=>{el.open=true;});
  const sent=page.waitForRequest(request=>request.url().endsWith('/audition'));
  const done=page.waitForResponse(response=>response.url().endsWith('/audition'));
  await page.locator('#auditionChart').click();
  expect((await sent).postDataJSON()).toMatchObject({mode:'chart',
    candidate_id:await page.evaluate(()=>variant().candidate_id),
    start_ms:Number(Math.max(0,event.start_ms/1000-1).toFixed(1))*1000});
  expect((await done).ok()).toBeTruthy();
  await page.locator('#auditionAudio').evaluate(el=>el.pause());
}

test('audio LN event wave locations use the saved candidate clock while transport keeps export time', async ({page})=>{
  const event=await openLNEvents(page,'audio');
  expect(await page.evaluate(()=>variant().audio_shift_ms)).toBe(22.5);
  await assertLocation(page,event,event.start_ms,event.start_ms+22.5);
  await page.locator('#waveClock').uncheck();
  await assertLocation(page,event,event.start_ms+22.5,event.start_ms+22.5);
  await page.locator('#audioShift').fill('50');await page.locator('#audioShift').blur();
  await page.locator('#waveClock').check();
  await assertLocation(page,event,event.start_ms-27.5,event.start_ms+22.5);
  await page.locator('#waveClock').uncheck();
  await assertLocation(page,event,event.start_ms+22.5,event.start_ms+22.5);
  await assertAuditionStart(page,event);
});

test('remap LN events have zero candidate shift even when the audio draft changes', async ({page})=>{
  const event=await openLNEvents(page,'remap');
  expect(await page.evaluate(()=>({mode:variant().mode,shift:variant().audio_shift_ms})))
    .toEqual({mode:'remap',shift:22.5});
  await assertLocation(page,event,event.start_ms,event.start_ms);
  await page.locator('#waveClock').uncheck();
  await assertLocation(page,event,event.start_ms,event.start_ms);
  await page.locator('#mode').selectOption('audio');
  await page.locator('#audioShift').fill('50');await page.locator('#audioShift').blur();
  await page.locator('#waveClock').check();
  await assertLocation(page,event,event.start_ms-50,event.start_ms);
  await page.locator('#waveClock').uncheck();
  await assertLocation(page,event,event.start_ms,event.start_ms);
  await assertAuditionStart(page,event);
});
