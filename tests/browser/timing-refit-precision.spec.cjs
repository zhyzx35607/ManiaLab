const { test, expect } = require('@playwright/test');

async function openTimingAdvanced(page) {
  const panel = page.locator('#timingAdvanced');
  if (!(await panel.evaluate(element => element.open))) {
    await panel.locator(':scope > summary').click();
  }
}

for(const entry of [
  {bpm:123.456789,offset:-37.5,normalized:-38,shift:22.5},
  {bpm:123.456789,offset:-1.5,normalized:-2,shift:-50},
  {bpm:120,offset:-37.5,normalized:-38,shift:0},
  {bpm:10,offset:-600000,normalized:-600000,shift:100},
  {bpm:1000,offset:600000,normalized:600000,shift:-100},
]){
  test(`fixed ${entry.bpm} BPM refit retains its diagnostic clock at shift ${entry.shift}`,async({page})=>{
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.goto('/');
    await expect(page.locator('#recent option')).toContainText(['UI timing clock']);
    await page.locator('#recent').selectOption({label:'UI timing clock'});
    await openTimingAdvanced(page);
    await expect(page.locator('#offset')).toHaveValue('93');
    await page.locator('#audioShift').fill(String(entry.shift));
    await page.locator('#audioShift').dispatchEvent('change');
    await page.locator('#bpm').fill(String(entry.bpm));
    await page.locator('#offset').fill(String(entry.offset));
    const pending=page.waitForResponse(response=>response.url().endsWith('/timing-refit'));
    await page.locator('#refitTiming').click();
    const response=await pending,report=await response.json();
    expect(response.status()).toBe(200);
    expect(response.request().postDataJSON().offset).toBe(entry.normalized);
    expect(report.bpm).toBe(entry.bpm);
    expect(report.clock).toEqual({bpm:entry.bpm,offset:report.offset,
      division:4,audio_shift_ms:entry.shift});
    await expect(page.locator('#bpm')).toHaveValue(String(entry.bpm));
    await expect(page.locator('#offset')).toHaveValue(String(report.offset));
    await expect(page.locator('#timingReport')).toContainText('重新估计 Offset');
    await expect.poll(()=>page.evaluate(()=>timingDiagnostics?.clock)).toEqual(report.clock);
    expect(errors).toEqual([]);
  });
}

for(const returnBeforeResponse of [false,true]){
test(`a mode ${returnBeforeResponse?'round trip':'change'} rejects an old refit and audio mode can request a fresh check`,async({page})=>{
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI LN navigation remap']);
  await page.locator('#recent').selectOption({label:'UI LN navigation remap'});
  await openTimingAdvanced(page);
  await expect(page.locator('#projectTitle')).toHaveText('UI LN navigation remap');
  await page.locator('#mode').selectOption('audio');
  await page.locator('#bpm').fill('120');
  await page.locator('#offset').fill('-37');
  await page.evaluate(()=>{
    const button=document.getElementById('refitTiming'),handler=button.onclick;
    button.onclick=(...args)=>{
      window.pendingTimingAction=handler(...args);
      return window.pendingTimingAction;
    };
  });
  let release,ready,oldResult;
  const gate=new Promise(resolve=>{release=resolve;});
  const prepared=new Promise(resolve=>{ready=resolve;});
  await page.route('**/timing-refit',async route=>{
    const response=await route.fetch();
    oldResult=await response.json();
    ready();await gate;
    await route.fulfill({response});
  });
  try{
    await page.locator('#refitTiming').click();
    await prepared;
    expect(oldResult.offset).not.toBe(-37);
    await page.locator('#mode').selectOption('remap');
    await expect(page.locator('#offset')).toBeDisabled();
    if(returnBeforeResponse)await page.locator('#mode').selectOption('audio');
    release();
    await page.evaluate(()=>window.pendingTimingAction);
    await expect(page.locator('#offset')).toHaveValue('-37');
    await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
    expect(await page.evaluate(()=>timingDiagnostics)).toBeNull();
  }finally{release();}
  await page.unroute('**/timing-refit');
  await page.locator('#mode').selectOption('audio');
  await expect(page.locator('#offset')).toHaveValue('-37');
  const pending=page.waitForResponse(response=>response.url().endsWith('/timing-refit'));
  await page.locator('#refitTiming').click();
  const response=await pending,report=await response.json();
  expect(response.status()).toBe(200);
  await expect(page.locator('#offset')).toHaveValue(String(report.offset));
  await expect(page.locator('#timingReport')).toContainText('重新估计 Offset');
  await expect.poll(()=>page.evaluate(()=>timingDiagnostics?.clock)).toEqual(report.clock);
  expect(errors).toEqual([]);
});
}
