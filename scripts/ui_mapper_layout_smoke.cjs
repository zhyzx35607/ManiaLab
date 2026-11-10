const {chromium}=require('../test-data/ui/node_modules/playwright');
const fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1450,height:1000}}),errors=[];
 page.setDefaultTimeout(20000);
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(process.argv[2]);await page.locator('#recent').selectOption(process.argv[3]);
  await page.waitForFunction(()=>state?.id);
  await page.locator('details').filter({has:page.locator('#engineGenerate')}).locator('summary').click();
  await page.locator('#engineUseTiming').uncheck();
  if(!await page.locator('#engineGenerate').textContent().then(t=>t.includes('保留模型键型')))throw Error('Button still promises reassignment');
  await page.locator('#engineGenerate').click({noWaitAfter:true});
  console.log('Synthetic model request submitted');
  try{await page.waitForFunction(()=>!busy&&variant()?.selection?.layout_origin==='mapperatorinator',{},{timeout:20000});}
  catch(e){console.log(await page.evaluate(()=>({busy,active:state?.active,v:variant(),message:document.querySelector('#message').textContent})));throw e;}
  const result=await page.evaluate(()=>({valid:variant().report.valid,lanes:variant().notes.map(n=>n.lane),body:document.body.textContent}));
  console.log(JSON.stringify({valid:result.valid,lanes:result.lanes,conflictVisible:result.body.includes('项结构冲突')}));
  if(result.valid||JSON.stringify(result.lanes)!=='[0,0]')throw Error('Original model layout was overwritten');
  if(!result.body.includes('项结构冲突')||result.body.includes('2 个物件，1 个 LN；结构检查通过'))throw Error('Conflict was reported as a pass');
  for(const id of ['exportOsu','exportOsz'])if(await page.locator('#'+id).getAttribute('href'))throw Error('Invalid chart export is still enabled');
  await page.screenshot({path:'tests/artifacts/model-layout-ui/result.png',fullPage:true,timeout:20000});
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync('tests/artifacts/model-layout-ui/report.json',JSON.stringify({passed:true,valid:result.valid,lanes:result.lanes,errors}));
  console.log(JSON.stringify({passed:true,valid:result.valid,lanes:result.lanes,errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
