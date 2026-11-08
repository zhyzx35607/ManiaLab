const {chromium}=require('../test-data/ui/node_modules/playwright');
const fs=require('fs');
(async()=>{
 const root='tests/artifacts/structure-app';
 const base=fs.readFileSync(root+'/current-url.txt','utf8').trim();
 const pid=fs.readFileSync(root+'/fixture-id.txt','utf8').trim();
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(base);await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>state?.id&&document.querySelectorAll('.structure-section').length>0);
  await page.locator('#bpm').fill('210');await page.locator('#offset').fill('93');await page.locator('#timingConfirmed').check();
  await page.locator('#structureBoundaries').fill('25, 60, 110, 165, 200');
  await page.locator('#sectionSave').click();
  await page.waitForFunction(()=>!busy&&state.analysis.structure.segments.length===6);
  if(await page.locator('#offset').inputValue()!=='93'||!await page.locator('#timingConfirmed').isChecked())throw Error('boundary edit changed confirmed timing');
  const row=page.locator('.structure-section').first();await row.locator('summary').click();
  await row.locator('[data-expression-voice=vocals]').selectOption('onsets');
  await page.locator('#expressionSave').click();
  await page.waitForFunction(()=>!busy&&state.expression.source==='manual');
  if(!await page.locator('#ln_ratio').isDisabled())throw Error('manual plan exposes misleading global quota');
  await page.locator('#generate').click();
  await page.waitForFunction(()=>!busy&&variant()?.selection?.expression_revision===1,{},{timeout:120000});
  const stats=await page.evaluate(()=>({notes:variant().notes.length,ln:variant().report.ln,
      decisions:variant().selection.decisions.length,grid:variant().report.grid.off_grid,
      invalid:!variant().report.valid,sections:state.analysis.structure.segments.length}));
  if(stats.grid||stats.invalid||!stats.decisions)throw Error('generation report failed');
  const exported=await page.request.get(base+`/api/projects/${pid}/export/osu`);
  const content=await exported.text();if(!content.includes('Bookmarks: 24978,59978,109978,164978,199978'))throw Error('export clock/bookmarks wrong');
  await page.reload();await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>state?.expression?.source==='manual');
  if(await page.evaluate(()=>state.expression.sections[0].voices.vocals.mode)!=='onsets')throw Error('source policy lost');
  await page.locator('.structure-section').first().locator('summary').click();
  await page.locator('#structurePanel').screenshot({path:'tests/artifacts/structure-desktop.png'});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  const result={...stats,errors,overflow,passed:!errors.length&&!overflow};
  fs.writeFileSync('tests/artifacts/structure-ui.json',JSON.stringify(result,null,2));console.log(result);
  if(!result.passed)throw Error('structure UI failed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
