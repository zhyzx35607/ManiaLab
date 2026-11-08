const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:5426';
 const waitJob=()=>page.waitForFunction(()=>!document.querySelector('#analyze').disabled,{},{timeout:900000});
 try{
  await page.goto(base);await page.waitForFunction(()=>!document.querySelector('#analysisEngine option[value=deep6]').disabled);
  if(process.env.MANIALAB_PROJECT_ID){
   await page.locator('#recent').selectOption(process.env.MANIALAB_PROJECT_ID);
   await page.waitForFunction(()=>document.querySelector('#analysisEngine').value==='deep6');
  }else{
   await page.locator('#files').setInputFiles(path.resolve('test-data/multistem/youdiantian.wav'));
   await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='youdiantian');
  }
  await page.locator('#analysisEngine').selectOption('deep6');await page.locator('#analyze').click();await waitJob();
  let state=await page.evaluate(()=>window.eval('state'));
  if(state?.analysis?.separation?.profile!=='six')throw Error(await page.locator('#message').textContent());
  if(state.analysis.voices.length!==6)throw Error('missing six-source inventory');
  for(const voice of ['guitar','piano']){
   await page.locator('#voice').selectOption(voice);
   const response=await page.request.get(base+`/api/projects/${state.id}/stem/${voice}`);
   if(response.status()!==200)throw Error('missing playable '+voice);
   const wave=await page.request.get(base+`/api/projects/${state.id}/waveform?voice=${voice}&start_ms=1000&duration_ms=500`);
   if(wave.status()!==200||(await wave.json()).sample_rate!==22050)throw Error('invalid waveform clock');
  }
  await page.locator('#bpm').fill('210');await page.locator('#offset').fill('93');await page.locator('#timingConfirmed').check();
  await page.locator('#guitar_weight').locator('xpath=ancestor::details[1]/summary').click();
  await page.locator('#guitar_weight').focus();await page.locator('#guitar_weight').press('Home');
  if(await page.locator('#guitar_weight').inputValue()!=='0')throw Error('guitar slider did not change');
  await page.locator('#generate').click();await waitJob();
  const v=await page.evaluate(()=>window.eval('variant()'));
  if(!v||v.report.grid.off_grid||v.notes.some(n=>n.voice==='guitar'))throw Error('voice choice/grid not honoured');
  if(!(await page.locator('#lnDensity button').count()))throw Error('missing LN occupancy timeline');
  const pid=state.id;await page.reload();await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>document.querySelector('#analysisEngine').value==='deep6');
  await page.screenshot({path:'tests/artifacts/multistem-workbench.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  const result={passed:!errors.length&&!overflow,voices:state.analysis.voices.map(v=>v.voice),notes:v.notes.length,offGrid:v.report.grid.off_grid,errors,overflow};
  fs.writeFileSync('tests/artifacts/multistem-ui.json',JSON.stringify(result,null,2));console.log(JSON.stringify(result));
  if(!result.passed)throw Error('UI failed');
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
