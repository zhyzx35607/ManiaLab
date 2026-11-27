const { test, expect } = require('@playwright/test');

function project(id, title, stable = true) {
  return {
    id, title, artist: 'Batch fixture', mode: 'audio', settings: { keys:10 },
    variants: [], active: -1, charts: [], audio: `${id}.wav`, original: null,
    input_notes: [], audio_shift_ms: 22.5,
    analysis: { revision:3, tempo:{ bpm:120, offset:93, division:4, stable } },
  };
}

const audioFile = name => ({
  name, mimeType:'audio/wav', buffer:Buffer.from('synthetic batch fixture'),
});

test('batch-wide rice preset requires timing acknowledgement and downloads successful candidates', async ({ page }) => {
  const ids = ['batch-onset-one', 'batch-onset-two', 'batch-onset-failed'];
  const titles = ['Onset Song One', 'Onset Song Two', 'Failed Song'];
  const projectState = new Map(ids.map((id,index) =>
    [id, project(id, titles[index], index !== 1)]));
  let analysisPoll = 0;
  let generationPoll = 0;
  let generationRequest = null;
  let generationSubmitted = false;
  let bundleItems = null;

  await page.route('**/api/import', async route => {
    const body = route.request().postData() || '';
    const index = ids.findIndex((id, i) => body.includes(`song-${i + 1}.wav`));
    await route.fulfill({ json:project(ids[index],titles[index],index!==1) });
  });
  await page.route('**/api/batch/analyze', async route => route.fulfill({ json:{
    id:'batch-analysis', status:'queued', batch:true, progress:0,
    items:ids.map((id,index)=>({project:id,title:titles[index],status:'queued',progress:0})),
  } }));
  await page.route('**/api/jobs/batch-analysis', async route => {
    analysisPoll++;
    await route.fulfill({ json:{
      id:'batch-analysis',status:'done',batch:true,progress:100,message:'批量分析完成',
      items:ids.map((id,index)=>index===2
        ?{project:id,title:titles[index],status:'error',progress:100,message:'合成分析失败'}
        :{project:id,title:titles[index],status:'done',progress:100,message:'完成',
          result:{candidates:12}}),
    } });
  });
  await page.route(/\/api\/projects\/batch-onset-(one|two|failed)$/, async route => {
    const id=new URL(route.request().url()).pathname.split('/').at(-1);
    await route.fulfill({json:projectState.get(id)});
  });
  await page.route('**/api/batch/generate', async route => {
    generationSubmitted=true;
    generationRequest=route.request().postDataJSON();
    await route.fulfill({json:{
      id:'batch-generation',status:'queued',batch:true,progress:0,
      items:generationRequest.projects.map(id=>({
        project:id,title:projectState.get(id).title,status:'queued',progress:0,
      })),
    }});
  });
  await page.route('**/api/jobs/batch-generation', async route => {
    generationPoll++;
    await route.fulfill({json:{
      id:'batch-generation',status:'done',batch:true,progress:100,message:'批量生成完成',
      items:[
        {project:ids[0],title:titles[0],status:'done',progress:100,result:{candidate_id:'a'.repeat(32),report:{ln:0}}},
        {project:ids[1],title:titles[1],status:'done',progress:100,result:{candidate_id:'b'.repeat(32),report:{ln:0}}},
      ],
    }});
  });

  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  await page.locator('#batchFiles').setInputFiles(ids.map((_,i)=>audioFile(`song-${i+1}.wav`)));
  await page.locator('#batchAnalyze').click();
  await expect(page.locator('#batchStatus')).toContainText('批次完成：2/3 首成功');
  await expect(page.locator('#batchGenerate')).toBeEnabled();
  await expect(page.locator('#batchGenerateControls')).toBeVisible();
  expect(analysisPoll).toBeGreaterThan(0);

  await page.locator('#batchKeys').selectOption('7');
  await expect(page.locator('#batchKeyLayout')).toHaveValue('7k_3_1_3');
  await page.locator('#batchEngine').selectOption('traditional');
  await page.locator('#batchPreset').selectOption('rice_stream');
  await expect(page.locator('#batchDensity')).toHaveValue('8');
  await page.locator('#batchDensity').fill('17');

  await page.locator('#batchGenerate').click();
  await expect(page.locator('#message')).toContainText('Onset Song Two');
  expect(generationSubmitted).toBe(false);

  await page.locator('#batchTimingConfirmed').check();
  await page.locator('#batchGenerate').click();
  await expect(page.locator('#batchStatus')).toContainText('统一生成结束：2/2 首成功');
  await expect(page.locator('#batchStatus')).toContainText('Failed Song');
  await expect(page.locator('#batchStatus [data-batch-download]')).toHaveCount(2);
  expect(generationPoll).toBeGreaterThan(0);
  expect(generationRequest).toMatchObject({
    projects:ids.slice(0,2),preset:'rice_stream',engine:'traditional',keys:7,key_layout:'7k_3_1_3',
    density:17,timing_confirmed:true,
  });
  expect(generationRequest.hold_note_ratio).toBeGreaterThan(0);
  const firstDownload=page.locator('#batchStatus [data-batch-download]').first();
  await expect(firstDownload).toHaveAttribute('href',new RegExp(`${ids[0]}.*candidate_id=${'a'.repeat(32)}`));

  await page.route('**/api/batch/export-osz?*', async route => {
    const url=new URL(route.request().url());
    bundleItems=url.searchParams.get('items');
    await route.fulfill({
      status:200,
      headers:{'content-type':'application/zip','content-disposition':'attachment; filename="ManiaLab-batch-osz.zip"'},
      body:Buffer.from('PK\\x03\\x04'),
    });
  });
  const downloadPromise=page.waitForEvent('download');
  await page.locator('#batchDownloadAll').click();
  const download=await downloadPromise;
  expect(await download.failure()).toBeNull();
  expect(bundleItems).toBe(`${ids[0]}:${'a'.repeat(32)},${ids[1]}:${'b'.repeat(32)}`);
});

test('changing to a different rice preset shows the existing candidate as stale and blocks chart export', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI onset stale']);
  await page.locator('#recent').selectOption({label:'UI onset stale'});
  await page.locator('#preset').selectOption('hybrid_adaptive');
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  const projectId=await page.locator('#recent').inputValue();
  const project=await (await page.request.get(`/api/projects/${projectId}`)).json();
  const candidate=project.variants[project.active];
  expect(candidate.report.ln).toBeGreaterThan(0);
  await expect(page.locator('#preset')).toHaveValue('hybrid_adaptive');
  await page.locator('#musical > .advanced-settings').evaluate(details=>{details.open=true;});
  await page.locator('#ln_ratio').fill('0.35');
  await expect(page.locator('#preset')).toHaveValue('custom');
  await expect(page.locator('#candidateDraftStatus')).toContainText('只用于下一次生成');
  await page.locator('#preset').selectOption('rice_stream');
  await expect(page.locator('#preset')).toHaveValue('rice_stream');
  await expect(page.locator('#candidateDraftStatus')).toContainText(
    `当前候选实际含 ${candidate.report.ln} 个 LN`,
  );
  for(const id of ['exportOsu','exportOsz']){
    await expect(page.locator('#'+id)).not.toHaveAttribute('href',/.+/);
    await expect(page.locator('#'+id)).toHaveClass(/disabled/);
  }
});

test('changing acquisition thresholds leaves the existing candidate exportable', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI acquisition settings']);
  await page.locator('#recent').selectOption({label:'UI acquisition settings'});
  await page.locator('#timingConfirmed').check();
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  await expect(page.locator('#exportOsu')).toHaveAttribute('href',/.+/);
  await expect(page.locator('#exportOsz')).toHaveAttribute('href',/.+/);

  await page.locator('.analysis-advanced').evaluate(details=>{details.open=true;});
  for(const [id,value] of [['onset_threshold','.5'],['frame_threshold','.35']]){
    await page.locator('#'+id).evaluate((input,next)=>{
      input.value=next;
      input.dispatchEvent(new Event('input',{bubbles:true}));
    },value);
  }

  await expect(page.locator('#candidateDraftStatus')).toBeHidden();
  await expect(page.locator('#exportOsu')).toHaveAttribute('href',/.+/);
  await expect(page.locator('#exportOsz')).toHaveAttribute('href',/.+/);
});

test('batch downloads cannot be activated until generation finishes', async ({ page }) => {
  const ids=['batch-download-one','batch-download-two'];
  const titles=['Download Song One','Download Song Two'];
  const projectState=new Map(ids.map((id,index)=>[id,project(id,titles[index])]));
  let analysisSnapshot=null;
  let generationPoll=0;
  let releaseGeneration;
  const finishGeneration=new Promise(resolve=>{releaseGeneration=resolve;});
  let individualDownloads=0;
  let bundleDownloads=0;

  await page.route('**/api/import',async route=>{
    const body=route.request().postData()||'';
    const index=ids.findIndex((_,i)=>body.includes(`download-${i+1}.wav`));
    await route.fulfill({json:project(ids[index],titles[index])});
  });
  await page.route('**/api/batch/analyze',async route=>route.fulfill({json:{
    id:'batch-download-analysis',status:'queued',batch:true,progress:0,
    items:ids.map((id,index)=>({project:id,title:titles[index],status:'queued',progress:0})),
  }}));
  await page.route('**/api/jobs/batch-download-analysis',async route=>{
    analysisSnapshot={id:'batch-download-analysis',status:'done',batch:true,progress:100,
      items:ids.map((id,index)=>({project:id,title:titles[index],status:'done',progress:100,
        result:{candidates:8}}))};
    await route.fulfill({json:analysisSnapshot});
  });
  await page.route(/\/api\/projects\/batch-download-(one|two)$/,async route=>{
    const id=new URL(route.request().url()).pathname.split('/').at(-1);
    await route.fulfill({json:projectState.get(id)});
  });
  await page.route('**/api/batch/generate',async route=>{
    await route.fulfill({json:{id:'batch-download-generation',status:'queued',batch:true,progress:0,
      items:ids.map((id,index)=>({project:id,title:titles[index],status:'queued',progress:0}))}});
  });
  await page.route('**/api/jobs/batch-download-generation',async route=>{
    generationPoll++;
    if(generationPoll===1){
      await route.fulfill({json:{id:'batch-download-generation',status:'running',batch:true,progress:50,
        items:[
          {project:ids[0],title:titles[0],status:'done',progress:100,result:{candidate_id:'c'.repeat(32)}},
          {project:ids[1],title:titles[1],status:'running',progress:25},
        ]}});
      return;
    }
    await finishGeneration;
    await route.fulfill({json:{id:'batch-download-generation',status:'done',batch:true,progress:100,
      items:ids.map((id,index)=>({project:id,title:titles[index],status:'done',progress:100,
        result:{candidate_id:(index===0?'c':'d').repeat(32)}}))}});
  });
  await page.route('**/api/batch/export-osz?*',async route=>{
    bundleDownloads++;
    await route.fulfill({status:200,headers:{'content-type':'application/zip'},body:Buffer.from('PK\\x03\\x04')});
  });
  await page.route(/\/api\/projects\/batch-download-(one|two)\/export\/osz\?/,async route=>{
    individualDownloads++;
    await route.fulfill({status:200,headers:{'content-type':'application/zip'},body:Buffer.from('PK\\x03\\x04')});
  });

  await page.goto('/');
  await page.locator('[data-work-mode="batch"]').click();
  await page.locator('#batchFiles').setInputFiles(ids.map((_,i)=>audioFile(`download-${i+1}.wav`)));
  await page.locator('#batchAnalyze').click();
  await expect(page.locator('#batchStatus')).toContainText('批次完成：2/2 首成功');
  await page.locator('#batchGenerate').click();

  const individual=page.locator('#batchStatus [data-batch-download]').first();
  const bundle=page.locator('#batchDownloadAll');
  await expect(individual).toBeVisible();
  await expect(individual).not.toHaveAttribute('href',/.+/);
  await expect(individual).toHaveAttribute('tabindex','-1');
  await expect(individual).toHaveAttribute('aria-disabled','true');
  await expect(bundle).not.toHaveAttribute('href',/.+/);
  await expect(bundle).toHaveAttribute('tabindex','-1');
  await expect(bundle).toHaveAttribute('aria-disabled','true');
  await individual.evaluate(link=>link.click());
  await bundle.evaluate(link=>link.click());
  expect(individualDownloads).toBe(0);
  expect(bundleDownloads).toBe(0);

  releaseGeneration();
  await expect(page.locator('#batchStatus')).toContainText('统一生成结束：2/2 首成功');
  await expect(individual).toHaveAttribute('href',new RegExp(`${ids[0]}.*candidate_id=${'c'.repeat(32)}`));
  await expect(individual).toHaveAttribute('aria-disabled','false');
  await expect(individual).not.toHaveAttribute('tabindex',/.+/);
  await expect(bundle).toHaveAttribute('href',/\/api\/batch\/export-osz\?/);
  await expect(bundle).toHaveAttribute('aria-disabled','false');
  await expect(bundle).not.toHaveAttribute('tabindex',/.+/);
  expect(generationPoll).toBeGreaterThan(1);
  expect(analysisSnapshot.items).toHaveLength(2);
});
