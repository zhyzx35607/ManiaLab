const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:5431';
 const audio=process.env.MANIALAB_STEM_FIXTURE||'test-data/sakura-benchmark/provided-ui';
 try{
  await page.goto(base);
  await page.locator('#files').setInputFiles(path.resolve(audio,'mix.wav'));
  await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='mix');
  await page.locator('#providedStemsPanel summary').click();
  for(const voice of ['drums','piano'])await page.locator('#provided-'+voice).setInputFiles(path.resolve(audio,voice+'.wav'));
  await page.locator('#providedStemsSave').click();
  await page.waitForFunction(()=>!document.querySelector('#analyze').disabled,{},{timeout:120000});
  if(await page.locator('#analysisEngine').inputValue()!=='provided')throw Error(await page.locator('#message').textContent());
  await page.locator('#analyze').click();
  if(!await page.locator('#providedStemsSave').isDisabled()||!await page.locator('#provided-piano').isDisabled())throw Error('Source import is editable during analysis');
  await page.waitForFunction(()=>!document.querySelector('#analyze').disabled,{},{timeout:300000});
  let state=await page.evaluate(()=>window.eval('state'));
  if(state?.analysis?.separation?.source!=='provided')throw Error(await page.locator('#message').textContent());
  if(state.analysis.voices.map(v=>v.voice).join()!=='drums,piano')throw Error('Unexpected source inventory');
  for(const voice of ['drums','piano']){
   await page.locator('#voice').selectOption(voice);
   const response=await page.request.get(`${base}/api/projects/${state.id}/stem/${voice}`);
   if(response.status()!==200)throw Error('Source preview missing: '+voice);
   const waveform=await page.request.get(`${base}/api/projects/${state.id}/waveform?voice=${voice}&start_ms=1000&duration_ms=500`);
   if(waveform.status()!==200||(await waveform.json()).sample_rate!==22050)throw Error('Source waveform clock mismatch');
  }
  await page.locator('#analyze').click();
  await page.waitForFunction(()=>!document.querySelector('#analyze').disabled,{},{timeout:300000});
  state=await page.evaluate(()=>window.eval('state'));
  if(!state.analysis.cached)throw Error('Imported pitch cache was not reused');
  const pid=state.id;
  await page.reload();await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>document.querySelector('#analysisEngine').value==='provided');
  await page.locator('#providedStemsPanel summary').click();
  await page.screenshot({path:'tests/artifacts/provided-stems-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  await page.screenshot({path:'tests/artifacts/provided-stems-mobile.png',fullPage:true});
  const report={passed:!errors.length&&!overflow,project:pid,voices:state.analysis.voices,
   candidates:state.analysis.candidates.length,cached:state.analysis.cached,overflow,errors};
  fs.writeFileSync('tests/artifacts/provided-stems-ui.json',JSON.stringify(report,null,2));console.log(JSON.stringify(report));
  if(!report.passed)throw Error('Provided-source UI failed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
