const { test, expect } = require('@playwright/test');

async function openProject(page, action) {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([`UI plan ${action}`]);
  await page.locator('#recent').selectOption({label:`UI plan ${action}`});
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await page.locator('#plan').evaluate(el=>{el.closest('details').open=true;});
  const pid=await page.locator('#recent').inputValue();
  return {pid,before:await (await page.request.get(`/api/projects/${pid}`)).json()};
}

async function draft(page) {
  await page.locator('#bpm').fill('137');await page.locator('#bpm').blur();
  await page.locator('#audioShift').fill('50');await page.locator('#audioShift').blur();
  await page.locator('#division').selectOption('8');
  await page.locator('.audition-panel').evaluate(el=>{el.open=true;});
  await page.locator('#auditionOffset').fill('27');
  await page.locator('#analysisEngine').selectOption('basic');
  await page.locator('#timingConfirmed').check();
}

async function assertDraft(page) {
  await expect(page.locator('#bpm')).toHaveValue('137');
  await expect(page.locator('#audioShift')).toHaveValue('50');
  await expect(page.locator('#offset')).toHaveValue('-49');
  await expect(page.locator('#division')).toHaveValue('8');
  await expect(page.locator('#auditionOffset')).toHaveValue('27');
  await expect(page.locator('#analysisEngine')).toHaveValue('basic');
  await expect(page.locator('#timingConfirmed')).toBeChecked();
  await expect(page.locator('#timingReport')).toContainText('当前拍线尚未检查');
  expect(await page.evaluate(()=>timingSections)).toEqual([]);
  await expect(page.locator('#exportOsu')).not.toHaveAttribute('href');
  await page.locator('#audioShift').fill('0');await page.locator('#audioShift').blur();
  await expect(page.locator('#offset')).toHaveValue('1');
}

async function requestPlan(page, model) {
  await page.locator('#endpoint').fill('fixture://planner');
  await page.locator('#model').fill(model);
  await page.locator('#instruction').fill('Use deterministic local fixture preferences');
  const queued=page.waitForResponse(response=>response.url().endsWith('/plan'));
  await page.locator('#plan').click();
  const response=await queued;expect(response.ok()).toBeTruthy();
  return (await response.json()).id;
}

test('clearing a plan preserves the draft clock and only removes the saved plan',async({page})=>{
  const {pid,before}=await openProject(page,'clear');
  await draft(page);
  const cleared=page.waitForResponse('**/clear-plan');
  await page.locator('#clearPlan').click();
  expect((await cleared).ok()).toBeTruthy();
  await expect(page.locator('#planText')).toHaveText('');
  const after=await (await page.request.get(`/api/projects/${pid}`)).json();
  const expected=structuredClone(before);delete expected.plan;
  expect(after).toEqual(expected);
  await assertDraft(page);
});

test('successful API planning preserves the draft and changes only planning preferences',async({page})=>{
  const {pid,before}=await openProject(page,'success');
  await draft(page);
  const jid=await requestPlan(page,'success');
  await expect(page.locator('#message')).toContainText('编排建议已应用');
  await expect(page.locator('#generate')).toBeEnabled();
  expect((await (await page.request.get(`/api/jobs/${jid}`)).json()).status).toBe('done');
  const after=await (await page.request.get(`/api/projects/${pid}`)).json();
  expect(after).toEqual({...before,plan:{preferences:{density:11},segments:[],explanation:'Deterministic fixture plan'},
    settings:{...before.settings,density:11}});
  await expect(page.locator('#density')).toHaveValue('11');
  await assertDraft(page);
});

for(const outcome of ['error','cancel'])test(`API plan ${outcome} preserves the previous plan and draft`,async({page})=>{
  const {pid,before}=await openProject(page,outcome);
  await draft(page);
  const jid=await requestPlan(page,outcome);
  if(outcome==='cancel'){
    await expect.poll(async()=> (await (await page.request.get('/__fixture/planner-state')).json()).waiting).toBe(true);
    const cancelled=page.waitForResponse(`**/api/jobs/${jid}/cancel`);
    await page.locator('#cancel').click();expect((await cancelled).ok()).toBeTruthy();
    const token=await page.locator('meta[name="manialab-token"]').getAttribute('content');
    expect((await page.request.post('/__fixture/planner-release',{headers:{'X-ManiaLab-Token':token}})).ok()).toBeTruthy();
    await expect(page.locator('#message')).toContainText('已取消');
  }else await expect(page.locator('#message')).toContainText('Deterministic fixture planner error');
  await expect(page.locator('#generate')).toBeEnabled();
  expect((await (await page.request.get(`/api/jobs/${jid}`)).json()).status).toBe(outcome==='cancel'?'cancelled':'error');
  expect(await (await page.request.get(`/api/projects/${pid}`)).json()).toEqual(before);
  await expect(page.locator('#planText')).toHaveText('Existing fixture plan');
  await assertDraft(page);
});

test('generation, reopening and import continue to adopt their own saved clocks',async({page})=>{
  const {pid}=await openProject(page,'scope');
  await draft(page);
  await page.locator('#generate').click();
  await expect(page.locator('#variants option')).toHaveCount(2);
  await expect(page.locator('#generate')).toBeEnabled();
  expect(await page.evaluate(()=>({bpm:variant().bpm,shift:variant().audio_shift_ms,division:variant().division})))
    .toEqual({bpm:137,shift:50,division:8});
  await page.locator('#bpm').fill('155');
  await page.locator('#audioShift').fill('0');await page.locator('#audioShift').blur();
  await page.locator('#recent').selectOption({label:'UI timing clock'});
  await expect(page.locator('#bpm')).toHaveValue('120');
  await expect(page.locator('#audioShift')).toHaveValue('22.5');
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#bpm')).toHaveValue('137');
  await expect(page.locator('#audioShift')).toHaveValue('50');
  const chart='osu file format v14\n\n[General]\nAudioFilename: song.wav\nMode: 3\n\n[Metadata]\nTitle: Clock import\nArtist: Fixture\nVersion: Test\n\n[Difficulty]\nCircleSize:10\n\n[TimingPoints]\n20,400,4,2,0,70,1,0\n\n[HitObjects]\n25,192,1003,128,0,2007:0:0:0:0:\n';
  await page.locator('#files').setInputFiles([{name:'clock.osu',mimeType:'text/plain',buffer:Buffer.from(chart)}]);
  await expect(page.locator('#projectTitle')).toHaveText('Clock import');
  await expect(page.locator('#mode')).toHaveValue('remap');
  await expect(page.locator('#bpm')).toHaveValue('120');
  await expect(page.locator('#audioShift')).toHaveValue('22.5');
  await expect(page.locator('#timingConfirmed')).not.toBeChecked();
});
