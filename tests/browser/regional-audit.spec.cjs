const { test, expect } = require('@playwright/test');

async function openProject(page, action) {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([`UI regional audit ${action}`]);
  await page.locator('#recent').selectOption({label:`UI regional audit ${action}`});
  await expect(page.locator('#projectTitle')).toHaveText(`UI regional audit ${action}`);
  await page.locator('#timingConfirmed').check();
}

async function selectNote(page, id, time) {
  await expect.poll(async()=>{
    const box=await page.evaluate(({id,time})=>{
      seek(time);draw();const hit=hitboxes.find(item=>item.n.id===id);
      return hit?{x:hit.x+hit.w/2,y:hit.y+hit.h/2}:null;
    },{id,time});
    if(!box)return false;
    await page.locator('#roll').click({position:box});
    return page.evaluate(id=>selected===id,id);
  }).toBe(true);
}

async function mutePiano(page) {
  await page.locator('#piano_weight').evaluate(el=>{
    el.value='0';el.dispatchEvent(new Event('input',{bubbles:true}));
  });
}

async function generateRegional(page, action) {
  await openProject(page,action);
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  const before=await page.evaluate(()=>structuredClone(variant().notes));
  expect(before.map(n=>n.id)).toEqual(['locked','region']);
  await mutePiano(page);
  await page.locator('#regenerate').evaluate(el=>{el.closest('details').open=true;});
  await page.locator('#regionStart').fill('3');
  await page.locator('#regionEnd').fill('5');
  await page.locator('#regenerate').click();
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  await expect(page.locator('#generate')).toBeEnabled();
  expect(await page.evaluate(()=>variant().notes)).toEqual(before);
  return page.locator('#recent').inputValue();
}

async function assertRegionalOutput(page) {
  const audit=page.locator('#expressionAudit');
  await expect(audit).toContainText('实际保留 2 个物件');
  await expect(audit).toContainText('区域外锁定 1 个');
  await expect(audit).toContainText('本次重写 1 个');
  await expect(audit).toContainText('实际 LN 比例 100%');
  await expect(audit).not.toContainText('当前候选的省音记录');
  await expect(audit).toContainText('区域外未采用这些新取舍');
  const proposal=audit.locator('details');
  await expect(proposal).not.toHaveAttribute('open');
  await proposal.locator('summary').click();
  const omitted=proposal.locator('button').filter({hasText:'钢琴'});
  await expect(omitted).toHaveCount(1);
  await expect(omitted).toContainText('同 ID 的旧物件已锁定保留，来源需分别核对');
  await expect(omitted).not.toContainText('实际已锁定保留');
  return omitted;
}

test('regional audit separates final locked output from whole-song proposed omissions',async({page})=>{
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  const pid=await generateRegional(page,'current');
  const current=await page.evaluate(()=>variant().selection);
  expect(current).toMatchObject({scope:'regional_output',selected:2,locked:1,generated:1,
    actual_ln_ratio:1,selected_voices:{piano:1,bass:1}});
  expect(current.decisions).toEqual(expect.arrayContaining([
    expect.objectContaining({id:'locked',decision:'selected',reason:'locked_existing',chart_start:1000}),
    expect.objectContaining({id:'region',decision:'selected',reason:'regional_regenerated'})]));
  expect(current.proposal).toMatchObject({scope:'full_song_proposal',selected:1});
  expect(current.proposal.decisions.find(n=>n.id==='locked')).toMatchObject({decision:'omitted',start:1000});
  const omitted=await assertRegionalOutput(page);
  await omitted.click();
  await expect.poll(()=>page.evaluate(()=>view)).toBe('evidence');
  await expect.poll(()=>page.evaluate(()=>currentTime())).toBeCloseTo(.6,2);
  await page.reload();await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  await assertRegionalOutput(page);
  expect(errors).toEqual([]);
});

test('editing a regional candidate keeps its origin and evidence without stale generation claims',async({page})=>{
  const pid=await generateRegional(page,'edit');
  const parent=await page.evaluate(()=>({index:state.active,candidate:structuredClone(variant())}));
  const originalEvidence=parent.candidate.source_evidence.region;
  await selectNote(page,'region',3.7);
  await page.locator('#noteEditor').evaluate(el=>{el.open=true;});
  const start=Number(await page.locator('#editStart').inputValue());
  await page.locator('#editStart').fill(String(start+125));
  await page.locator('#editSave').click();

  await expect(page.locator('#variants option:checked')).toContainText('编辑自局部候选');
  await expect(page.locator('#inspector')).toContainText('人工修改');
  await expect(page.locator('#inspector')).toContainText('原始候选起点（原音频）');
  await expect(page.locator('#inspector')).toContainText('来源描述保留原证据，不为人工修改后的音尾背书');
  const edited=await page.evaluate(()=>structuredClone(variant()));
  expect(edited.region).toBeNull();
  expect(edited.generation_origin).toEqual({kind:'regional_generation',region:[3000,5000]});
  expect(edited.source_evidence.region).toEqual(originalEvidence);
  expect(edited.edits.at(-1)).toMatchObject({
    op:'update',parent_created:parent.candidate.created,before:{id:'region'},after:{id:'region'},
  });
  expect(edited.selection).not.toHaveProperty('scope');
  expect(edited.selection).not.toHaveProperty('proposal');
  const audit=page.locator('#expressionAudit');
  await expect(audit).not.toContainText('省略提案');
  await expect(audit).not.toContainText('区域外锁定');
  await expect(audit).not.toContainText('本次重写');

  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#variants option:checked')).toContainText('编辑自局部候选');
  const editedIndex=await page.evaluate(()=>state.active);
  await page.locator('#variants').selectOption(String(parent.index));
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  await assertRegionalOutput(page);
  await page.locator('#variants').selectOption(String(editedIndex));
  await expect(page.locator('#variants option:checked')).toContainText('编辑自局部候选');
  await expect(page.locator('#expressionAudit')).not.toContainText('区域外锁定');
  await selectNote(page,'region',3.7);
  await expect(page.locator('#inspector')).toContainText('人工修改');
  await expect(page.locator('#inspector')).toContainText('原始候选起点（原音频）');

  await page.locator('#noteEditor').evaluate(el=>{el.open=true;});
  const secondStart=Number(await page.locator('#editStart').inputValue());
  const secondEnd=Number(await page.locator('#editEnd').inputValue());
  await page.locator('#editStart').fill(String(secondStart+125));
  await page.locator('#editEnd').fill(String(secondEnd+125));
  await page.locator('#editSave').click();
  await expect(page.locator('#variants option:checked')).toContainText('编辑自局部候选');
  const twiceEdited=await page.evaluate(()=>structuredClone(variant()));
  expect(twiceEdited.generation_origin).toEqual(edited.generation_origin);
  expect(twiceEdited.edits).toHaveLength(2);
  expect(twiceEdited.edits.at(-1).parent_created).toBe(edited.created);
  expect(twiceEdited.source_evidence.region).toEqual(originalEvidence);
  expect(twiceEdited.selection).not.toHaveProperty('proposal');
  await expect(page.locator('#expressionAudit')).not.toContainText('省略提案');

  await page.route(`**/api/projects/${pid}`,async route=>{
    const response=await route.fetch(),data=await response.json();
    data.variants=[data.variants[data.active]];data.active=0;
    await route.fulfill({response,json:data});
  });
  await page.reload();
  await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#variants option')).toHaveCount(1);
  await expect(page.locator('#variants option:checked')).toContainText('编辑自局部候选');
  const trimmed=await page.evaluate(()=>structuredClone(variant()));
  expect(trimmed.generation_origin).toEqual({kind:'regional_generation',region:[3000,5000]});
  expect(trimmed.selection).not.toHaveProperty('proposal');
  await expect(page.locator('#expressionAudit')).not.toContainText('区域外锁定');
  await expect(page.locator('#expressionAudit')).not.toContainText('本次重写');
  const currentStart=await page.evaluate(()=>variant().notes.find(note=>note.id==='region').start);
  await selectNote(page,'region',(currentStart+100)/1000);
  await expect(page.locator('#inspector')).toContainText('人工修改');
  await expect(page.locator('#inspector')).toContainText('原始候选起点（原音频）');
});

test('legacy regional reports derive counts from saved notes and label their uncertain scope',async({page})=>{
  const pid=await generateRegional(page,'legacy');
  await page.route(`**/api/projects/${pid}`,async route=>{
    const response=await route.fetch(),data=await response.json(),v=data.variants[data.active];
    v.selection=v.selection.proposal;delete v.selection.scope;
    await route.fulfill({response,json:data});
  });
  await page.reload();await page.locator('#recent').selectOption(pid);
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  expect(await page.evaluate(()=>variant().selection.selected)).toBe(1);
  await expect(page.locator('#expressionAudit')).toContainText('旧候选未记录取舍范围');
  await expect(page.locator('#expressionAudit')).toContainText('整首预选参考，不能代表最终省音');
  await assertRegionalOutput(page);
});

test('whole-song generation keeps the existing final omission audit',async({page})=>{
  await openProject(page,'full');await mutePiano(page);
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  expect(await page.evaluate(()=>variant().notes.map(n=>n.id))).toEqual(['region']);
  await expect(page.locator('#expressionAudit')).toContainText('当前候选的省音记录');
  await expect(page.locator('#expressionAudit')).not.toContainText('区域外锁定');
  await page.locator('#expressionAudit summary').click();
  await expect(page.locator('#expressionAudit button')).toContainText('钢琴');
  await expect(page.locator('#expressionAudit button')).not.toContainText('实际已锁定保留');
});

test('reanalyzed reused IDs identify the old lock and locate only the new omitted proposal',async({page})=>{
  await openProject(page,'reanalyzed');
  await expect(page.locator('#variants option')).toHaveCount(1);
  const before=await page.evaluate(()=>structuredClone(variant().notes));
  expect(before.find(n=>n.id==='locked')).toMatchObject({start:1000,end:1750});
  await mutePiano(page);
  await page.locator('#regenerate').evaluate(el=>{el.closest('details').open=true;});
  await page.locator('#regionStart').fill('3');await page.locator('#regionEnd').fill('5');
  await page.locator('#regenerate').click();
  await expect(page.locator('#variants option:checked')).toContainText('局部');
  await expect(page.locator('#generate')).toBeEnabled();
  const current=await page.evaluate(()=>structuredClone(variant()));
  expect(current.notes).toEqual(before);
  expect(current.selection.decisions.find(n=>n.id==='locked')).toMatchObject({reason:'locked_existing',chart_start:1000});
  expect(current.selection.proposal.decisions.find(n=>n.id==='locked')).toMatchObject({decision:'omitted',start:3000,end:3300});
  expect(current.source_evidence.locked.detected).toMatchObject({start:1000,end:1800});
  const omitted=await assertRegionalOutput(page);
  await expect(omitted).toContainText('提案 00:03.000');
  await expect(omitted).not.toContainText('实际已保留');
  await omitted.click();
  await expect.poll(()=>page.evaluate(()=>view)).toBe('evidence');
  await expect.poll(()=>page.evaluate(()=>currentTime())).toBeCloseTo(2.6,2);
  expect(await page.evaluate(()=>variant().notes.find(n=>n.id==='locked').start)).toBe(1000);
});
