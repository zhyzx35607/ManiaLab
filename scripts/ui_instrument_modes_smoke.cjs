const {chromium}=require('../test-data/ui/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const base=process.argv[2],pid=process.argv[3],output=process.argv[4];
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1450,height:1000}}),errors=[];
 page.setDefaultTimeout(20000);page.on('pageerror',e=>errors.push(e.message));
 const controls=section=>page.locator(`.structure-section`).nth(section);
 const instrument=(section,program)=>controls(section).locator(`[data-expression-instrument="yourmt3:program:${program}"]`);
 async function openFirst(){
  await controls(0).evaluate(e=>e.open=true);
  await controls(0).locator('.instrument-modes').evaluate(e=>e.open=true);
 }
 async function save(){
  const response=page.waitForResponse(r=>r.url().endsWith(`/api/projects/${pid}/expression`)&&r.request().method()==='POST');
  await page.locator('#expressionSave').click();
  if((await response).status()!==200)throw Error('Saving instrument modes failed');
  await page.waitForFunction(()=>!busy);
 }
 async function generate(){
  const previous=await page.evaluate(()=>state.variants.length);
  await page.locator('#generate').click();
  await page.waitForFunction(n=>!busy&&state.variants.length>n,previous,{timeout:30000});
  return page.evaluate(()=>({notes:variant().notes,decisions:variant().selection.decisions,report:variant().report}));
 }
 try{
  await page.goto(base);await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>state?.id&&document.querySelectorAll('.structure-section').length===2);
  await openFirst();
  if(await page.locator('[data-expression-instrument]').count()!==3)throw Error('Invented a class for unknown evidence or omitted a known class');
  if(await instrument(0,48).inputValue()!=='inherit')throw Error('Legacy plan did not inherit');
  await instrument(0,48).selectOption('sustain');await instrument(0,24).selectOption('omit');
  await save();
  await page.reload();await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(()=>state?.expression?.source==='manual');await openFirst();
  if(await instrument(0,48).inputValue()!=='sustain'||await instrument(0,24).inputValue()!=='omit')throw Error('Mode did not survive project reload');
  if(await instrument(1,48).inputValue()!=='inherit')throw Error('Override leaked into a later section');
  await page.locator('#audioShift').fill('0');await page.locator('#bpm').fill('120');await page.locator('#offset').fill('0');await page.locator('#timingConfirmed').check();
  const first=await generate();
  const notes=Object.fromEntries(first.notes.map(n=>[n.id,n]));
  if(notes.strings.end!==2500||notes.unknown.end!==3000||notes.later.end!==8000||notes.guitar)throw Error('Effective modes do not reach generated chart');
  if(!first.report.valid||first.report.grid.off_grid)throw Error('Invalid generated chart');
  await openFirst();await controls(0).locator('[data-expression-voice="other"]').selectOption('omit');await save();
  const second=await generate();
  if(!second.notes.some(n=>n.id==='strings'&&n.end===2500)||second.notes.some(n=>n.id==='unknown'))throw Error('Explicit class cannot reopen omitted voice or unknown did not inherit');
  await openFirst();await page.locator('#structurePanel').screenshot({path:path.join(output,'desktop.png')});
  await page.setViewportSize({width:390,height:844});
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  await page.locator('#structurePanel').screenshot({path:path.join(output,'mobile.png')});
  if(overflow||errors.length)throw Error('UI overflow or JavaScript error: '+JSON.stringify({overflow,errors}));
  await instrument(0,48).selectOption('inherit');await save();
  const third=await generate();
  if(third.notes.some(n=>n.id==='strings'))throw Error('Inherit did not restore voice omission');
  const report={passed:true,synthetic_fixture:true,musical_acceptance:false,errors,overflow,
   first:first.notes.map(n=>({id:n.id,start:n.start,end:n.end,instrument_id:n.instrument_id})),
   explicit_override:second.notes.map(n=>n.id),restored_inheritance:third.notes.map(n=>n.id)};
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
