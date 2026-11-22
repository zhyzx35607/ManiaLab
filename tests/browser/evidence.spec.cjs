const { test, expect } = require('@playwright/test');

async function generateEvidence(page, title) {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText([title]);
  await page.locator('#recent').selectOption({ label:title });
  await expect(page.locator('#projectTitle')).toHaveText(title);
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await page.locator('#reviewSummary').evaluate(el => { el.closest('details').open = true; });
}

async function selectNote(page, id, time) {
  await expect.poll(async()=>{
    const box=await page.evaluate(({id,time})=>{
      seek(time);draw();const b=hitboxes.find(b=>b.n.id===id);
      return b?{x:b.x+b.w/2,y:b.y+b.h/2}:null;
    },{id,time});
    if(!box)return false;
    await page.locator('#roll').click({position:box});
    return page.evaluate(id=>selected===id,id);
  }).toBe(true);
}

async function reopenProject(page,pid){
  await page.reload();await page.locator('#recent').selectOption(pid);
  await expect.poll(()=>page.evaluate(pid=>state?.id===pid&&!!variant(),pid)).toBe(true);
  await expect(page.locator('#gridReport')).toContainText('落线检查');
}

function peakTime(data) {
  const index=data.high.indexOf(Math.max(...data.high));
  return (data.first_sample+index*data.step_samples)*1000/data.sample_rate;
}

test('selected candidate keeps its voice source while raw evidence uses the current analysis', async ({page})=>{
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI evidence source identity']);
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await expect(page.locator('#variants option:checked')).toContainText('候选 1');
  await page.locator('#voice').selectOption('other');
  await expect.poll(()=>page.locator('#audio').evaluate(el=>new URL(el.src).searchParams.get('context'))).toBe('chart');
  await page.locator('#waveDetailPanel').evaluate(el=>{el.open=true;});

  const chartResponse=page.waitForResponse(response=>response.url().includes('/waveform?')&&response.url().includes('context=chart'));
  await page.locator('#waveHere').click();
  const chart=await (await chartResponse).json();
  expect(Math.abs(peakTime(chart)-100)).toBeLessThan(15);

  await page.locator('#evidenceView').click();
  await expect.poll(()=>page.locator('#audio').evaluate(el=>new URL(el.src).searchParams.get('context'))).toBe('evidence');
  const evidenceResponse=page.waitForResponse(response=>response.url().includes('/waveform?')&&response.url().includes('context=evidence'));
  await page.locator('#waveHere').click();
  const evidence=await (await evidenceResponse).json();
  expect(Math.abs(peakTime(evidence)-600)).toBeLessThan(15);
});

test('audition buttons adopt the source of the requested review mode', async ({page})=>{
  await page.goto('/');
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await expect(page.locator('#projectTitle')).toHaveText('UI evidence source identity');
  await page.locator('#voice').selectOption('other');
  await page.locator('.audition-panel').evaluate(panel=>{panel.open=true;});
  for(const mode of ['evidence','chart']){
    const responsePromise=page.waitForResponse(response=>response.url().endsWith('/audition'));
    await page.locator(mode==='chart'?'#auditionChart':'#auditionEvidence').click();
    const response=await responsePromise,payload=response.request().postDataJSON();
    expect(response.status()).toBe(200);
    expect(payload.context).toBe(mode);
    expect(payload.mode).toBe(mode);
    expect(payload.voice).toBe('other');
    const identity=await page.evaluate(mode=>mode==='chart'?
      variant().source_separation.identity:state.analysis.separation.playback_identity,mode);
    expect(payload.source_identity).toBe(identity);
    expect(payload.candidate_id).toBe(mode==='chart'?
      await page.evaluate(()=>variant().candidate_id):undefined);
    await expect(page.locator(mode==='chart'?'#chartView':'#evidenceView')).toHaveClass('active');
  }
});

test('waveform response from an old review context cannot replace the current source', async ({page})=>{
  await page.goto('/');
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await expect(page.locator('#projectTitle')).toHaveText('UI evidence source identity');
  await page.locator('#voice').selectOption('other');
  await page.locator('#waveDetailPanel').evaluate(el=>{el.open=true;});
  let releaseChart;
  const chartGate=new Promise(resolve=>{releaseChart=resolve;});
  await page.route('**/waveform?*',async route=>{
    const context=new URL(route.request().url()).searchParams.get('context');
    if(context==='chart')await chartGate;
    await route.continue();
  });
  const staleResponse=page.waitForResponse(response=>response.url().includes('/waveform?')&&response.url().includes('context=chart'));
  await page.locator('#waveHere').click();
  const evidenceResponse=page.waitForResponse(response=>response.url().includes('/waveform?')&&response.url().includes('context=evidence'));
  await page.locator('#evidenceView').click();
  const evidence=await (await evidenceResponse).json();
  releaseChart();
  await staleResponse;
  expect(Math.abs(peakTime(evidence)-600)).toBeLessThan(15);
  const finalPeak=await page.evaluate(()=>{
    const data=waveDetail,index=data.high.indexOf(Math.max(...data.high));
    return (data.first_sample+index*data.step_samples)*1000/data.sample_rate;
  });
  expect(Math.abs(finalPeak-600)).toBeLessThan(15);
});

test('mix playback keeps its source and playhead when only the review view changes', async ({page})=>{
  await page.goto('/');
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await page.locator('#voice').selectOption('mix');
  const srcBefore=await page.locator('#audio').evaluate(el=>el.getAttribute('src'));
  await page.locator('#audio').evaluate(el=>{el.currentTime=3;});
  await page.locator('#evidenceView').click();
  await expect(page.locator('#audio')).toHaveAttribute('src',srcBefore);
  await expect.poll(()=>page.locator('#audio').evaluate(el=>el.currentTime)).toBeCloseTo(3,1);
});

test('switching review context discards an in-flight audition from the previous context', async ({page})=>{
  await page.goto('/');
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await page.locator('.audition-panel').evaluate(panel=>{panel.open=true;});
  let releaseResponse;
  const responseGate=new Promise(resolve=>{releaseResponse=resolve;});
  await page.route('**/audition',async route=>{
    await responseGate;
    await route.fulfill({status:200,contentType:'audio/wav',body:'stale audition'});
  });
  const request=page.waitForRequest(req=>req.url().endsWith('/audition')&&req.method()==='POST');
  await page.locator('#auditionChart').click();
  await request;
  await page.locator('#evidenceView').click();
  releaseResponse();
  await page.waitForResponse(res=>res.url().endsWith('/audition'));
  await expect.poll(()=>page.locator('#auditionAudio').evaluate(player=>player.getAttribute('src'))).toBeNull();
});

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

test('deleting a selected note creates a reversible candidate that survives reopening', async ({ page }) => {
  await generateEvidence(page,'UI evidence edits');
  const pid=await page.locator('#recent').inputValue();
  const original=await page.evaluate(()=>({
    created:variant().created,
    count:variant().notes.length,
  }));
  await selectNote(page,'unobserved',8.8);
  await page.locator('#noteEditor').evaluate(el=>{el.open=true;});
  await page.locator('#editDelete').click();
  await expect.poll(()=>page.evaluate(()=>variant().notes.some(note=>note.id==='unobserved'))).toBe(false);
  await expect(page.locator('#editDelete')).toBeDisabled();

  const deleted=await page.evaluate(()=>state);
  const originalIndex=deleted.variants.findIndex(candidate=>candidate.created===original.created);
  const deletedIndex=deleted.active;
  expect(originalIndex).toBeGreaterThanOrEqual(0);
  expect(deleted.variants[originalIndex].notes).toHaveLength(original.count);
  expect(deleted.variants[deletedIndex].notes).toHaveLength(original.count-1);
  expect(deleted.variants[deletedIndex].edits.at(-1)).toMatchObject({
    op:'delete',
    parent_created:original.created,
    before:{id:'unobserved'},
    after:null,
  });

  await page.locator('#variants').selectOption(String(originalIndex));
  await expect.poll(()=>page.evaluate(()=>variant().notes.some(note=>note.id==='unobserved'))).toBe(true);
  await page.locator('#variants').selectOption(String(deletedIndex));
  await expect.poll(()=>page.evaluate(()=>variant().notes.some(note=>note.id==='unobserved'))).toBe(false);
  await reopenProject(page,pid);
  await expect.poll(()=>page.evaluate(()=>variant().notes.some(note=>note.id==='unobserved'))).toBe(false);
  await page.locator('#variants').selectOption(String(originalIndex));
  await expect.poll(()=>page.evaluate(()=>variant().notes.some(note=>note.id==='unobserved'))).toBe(true);
});

test('deleted note source snapshots remain historical in JSON and waveform', async ({ page }) => {
  await generateEvidence(page,'UI evidence edits');
  const pid=await page.locator('#recent').inputValue();
  const original=await page.evaluate(()=>({
    sourceEvidence:structuredClone(variant().source_evidence.evidence0),
    noteStart:variant().notes.find(note=>note.id==='evidence0').start/1000,
  }));
  await selectNote(page,'evidence0',original.noteStart);
  await page.locator('#noteEditor').evaluate(el=>{el.open=true;});
  await page.locator('#editDelete').click();
  await expect.poll(()=>page.evaluate(()=>variant().edits?.at(-1)?.op)).toBe('delete');

  const exported=await (await page.request.get(`/api/projects/${pid}/export/json`)).json();
  expect(exported.variant.notes.some(note=>note.id==='evidence0')).toBe(false);
  expect(exported.variant.source_evidence.evidence0).toEqual(original.sourceEvidence);
  expect(exported.variant.edits.at(-1)).toMatchObject({op:'delete',before:{id:'evidence0'},after:null});

  await page.locator('#voice').selectOption('other');
  await page.locator('#waveDetailPanel').evaluate(el=>{el.open=true;});
  const onsetTimes=await page.evaluate(()=>({
    deleted:waveAudioTime(variant().source_evidence.evidence0.detected.start),
    current:waveAudioTime(variant().source_evidence.evidence1.detected.start),
  }));
  await page.evaluate(time=>loadWaveDetail(time),(onsetTimes.deleted+onsetTimes.current)/2);
  await expect(page.locator('#waveDetailInfo')).toContainText(
    '当前谱面中有已保存快照物件的原始起音');
  await expect(page.locator('#waveDetailInfo')).toContainText('不在当前谱面的历史起音');
  const markers=await page.evaluate(()=>{
    const canvas=document.getElementById('waveDetail'),context=canvas.getContext('2d'),records=[];
    const move=context.moveTo;
    context.moveTo=function(x,y){
      if(this.strokeStyle==='#ffb978'||this.strokeStyle==='#c88d70')
        records.push({color:this.strokeStyle,x,dash:this.getLineDash()});
      return move.call(this,x,y);
    };
    try{drawWaveDetail();}finally{context.moveTo=move;}
    const width=canvas.getBoundingClientRect().width-24;
    return records.map(record=>({...record,time:waveDetail.start_ms+(record.x-12)/width*(waveDetail.end_ms-waveDetail.start_ms)}));
  });
  expect(markers.some(marker=>marker.color==='#c88d70'&&marker.dash.length>0&&
    Math.abs(marker.time-onsetTimes.deleted)<1)).toBe(true);
  expect(markers.some(marker=>marker.color==='#ffb978'&&marker.dash.length===0&&
    Math.abs(marker.time-onsetTimes.deleted)<1)).toBe(false);
  expect(markers.some(marker=>marker.color==='#ffb978'&&marker.dash.length===0&&
    Math.abs(marker.time-onsetTimes.current)<1)).toBe(true);
  expect(markers.some(marker=>marker.color==='#c88d70'&&marker.dash.length>0&&
    Math.abs(marker.time-onsetTimes.current)<1)).toBe(false);
});

test('chart waveform legend only describes saved source snapshots', async ({ page }) => {
  await generateEvidence(page,'UI evidence legend');
  const snapshot=await page.evaluate(()=>structuredClone(variant().source_evidence.evidence0));
  await page.locator('#voice').selectOption('other');
  await page.evaluate(()=>{delete variant().source_evidence;});
  await page.locator('#waveDetailPanel').evaluate(el=>{el.open=true;});
  await page.evaluate(()=>loadWaveDetail(1000));
  await expect(page.locator('#waveDetailInfo')).toContainText('此候选未保存可用的原始起音快照');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('橙色：');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('灰橙虚线：');

  await page.evaluate(snapshot=>{
    variant().source_evidence={evidence0:snapshot};
    drawWaveDetail();
  },snapshot);
  await expect(page.locator('#waveDetailInfo')).toContainText(
    '橙色：当前谱面中有已保存快照物件的原始起音');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('灰橙虚线：');

  await page.locator('#voice').selectOption('piano');
  await expect(page.locator('#waveDetailInfo')).toContainText('当前声部没有可用的已保存原始起音快照');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('橙色：');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('灰橙虚线：');

  await page.evaluate(()=>{state.active=-1;drawWaveDetail();});
  await expect(page.locator('#waveDetailInfo')).toContainText('尚未生成候选谱面');
  await expect(page.locator('#waveDetailInfo')).not.toContainText('此候选未保存');
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
    await reopenProject(page,pid);
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
  await expect(page.locator('#projectTitle')).toHaveText('UI evidence legacy region');
  await expect.poll(()=>page.evaluate(()=>variant()?.notes.some(n=>n.id==='evidence0'))).toBe(true);
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
  await reopenProject(page,pid);
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
