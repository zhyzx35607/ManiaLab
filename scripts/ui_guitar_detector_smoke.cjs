const {chromium}=require('../test-data/ui/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('assert');

(async()=>{
 const [base,output,fixture]=process.argv.slice(2);
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1450,height:1000}}),errors=[],requests=[],checks=[];
 page.setDefaultTimeout(20000);
 page.on('pageerror',e=>errors.push(e.message));
 page.on('request',r=>{if(r.method()==='POST'&&/\/api\/projects\/[^/]+\/analyze$/.test(new URL(r.url()).pathname)){
  const data=r.postDataJSON();requests.push(Object.fromEntries(['engine','guitar_engine','allow_guitar_predicted_releases'].map(k=>[k,data[k]])));
 }});
 const main=page.locator('#guitarSpecialist'),tail=page.locator('#guitarPredictedReleases');
 const selected=()=>page.evaluate(()=>window.guitarSelection());
 const general=engine=>({engine,guitar_engine:'general',allow_guitar_predicted_releases:false});
 const specialized=(engine,releases)=>({engine,guitar_engine:'fl',allow_guitar_predicted_releases:releases});
 let overflow=null;
 async function assertControls(enabled,checked,releases){
  assert.strictEqual(await main.isEnabled(),enabled);
  assert.strictEqual(await main.isChecked(),checked);
  assert.strictEqual(await tail.isChecked(),releases);
  assert.strictEqual(await tail.isEnabled(),enabled&&checked);
 }
 async function analyze(expected,ending='done'){
  const count=requests.length;
  await page.locator('#analyze').click();
  await page.waitForFunction(()=>busy&&jobId!==null);
  assert(await main.isDisabled()&&await tail.isDisabled(),'Controls changed during analysis');
  if(ending==='cancelled')await page.locator('#cancel').click();
  await page.waitForFunction(()=>!busy);
  assert.strictEqual(requests.length,count+1);
  assert.deepStrictEqual(requests.at(-1),expected);
  const text=await page.locator('#message').textContent();
  if(ending==='error')assert(text.includes('模拟采音失败'),text);
  else if(ending==='cancelled')assert(text.includes('取消'),text);
  else assert(text.includes('已提取'),text);
 }
 async function reopen(pid){
  await page.reload();
  await page.waitForFunction(id=>[...document.querySelector('#recent').options].some(o=>o.value===id),pid);
  await page.locator('#recent').selectOption(pid);
  await page.waitForFunction(id=>state?.id===id&&!busy,pid);
 }
 async function importStem(voice){
  await page.locator('#providedStemsPanel').evaluate(e=>e.open=true);
  await page.locator('#provided-'+voice).setInputFiles(fixture);
  await page.locator('#providedStemsSave').click();
  await page.waitForFunction(v=>!busy&&state?.provided_stems?.voices.join()===v&&document.querySelector('#analysisEngine').value==='provided',voice);
 }
 try{
  await page.goto(base);
  await page.waitForFunction(()=>document.querySelector('#analysisEngine').value==='deep'&&
   document.querySelector('#guitarDetector').textContent.includes('选择六声部分离'));
  await assertControls(false,false,false);
  assert.deepStrictEqual(await selected(),{guitar_engine:'general',allow_guitar_predicted_releases:false});
  checks.push('initial_deep_disabled_and_default_off');
  await page.locator('#files').setInputFiles(fixture);
  await page.waitForFunction(()=>state?.audio&&!busy);
  const pid=await page.evaluate(()=>state.id);
  await analyze(general('deep'));checks.push('deep_request_stays_general');
  await page.locator('#analysisEngine').selectOption('deep6');
  await assertControls(true,false,false);
  await analyze(general('deep6'));checks.push('six_stem_eligible_default_off');
  await main.check();await assertControls(true,true,false);
  await analyze(specialized('deep6',false));await reopen(pid);
  await assertControls(true,true,false);checks.push('onsets_only_project_restores');
  await tail.check();await analyze(specialized('deep6',true));await reopen(pid);
  await assertControls(true,true,true);checks.push('predicted_tails_project_restores');
  await page.locator('#analysisEngine').selectOption('deep');
  await assertControls(false,true,true);
  assert.deepStrictEqual(await selected(),{guitar_engine:'general',allow_guitar_predicted_releases:false});
  await analyze(general('deep'));await assertControls(false,false,false);checks.push('deep_excludes_saved_specialist');
  await page.locator('#analysisEngine').selectOption('deep6');await main.check();await tail.check();
  await page.locator('#analysisEngine').selectOption('basic');await assertControls(false,true,true);
  await analyze(general('basic'));await assertControls(false,false,false);checks.push('basic_excludes_specialist');
  await importStem('guitar');await assertControls(true,false,false);checks.push('provided_guitar_eligible');
  await main.check();await tail.check();await analyze(specialized('provided',true));await reopen(pid);
  await assertControls(true,true,true);checks.push('provided_guitar_project_restores');
  await main.uncheck();await assertControls(true,false,false);checks.push('parent_toggle_clears_tails');
  await main.check();await tail.check();
  const saved=await page.evaluate(()=>JSON.stringify(state));
  await page.evaluate(()=>api('/__smoke/mode',{mode:'cancel'}));
  await analyze(specialized('provided',true),'cancelled');
  assert.strictEqual(await page.evaluate(()=>JSON.stringify(state)),saved);
  await assertControls(true,true,true);
  await page.evaluate(()=>api('/__smoke/mode',{mode:'error'}));
  await analyze(specialized('provided',true),'error');
  assert.strictEqual(await page.evaluate(()=>JSON.stringify(state)),saved);
  await assertControls(true,true,true);
  await page.evaluate(()=>api('/__smoke/mode',{mode:'success'}));checks.push('busy_cancel_failure_restore_controls_and_preserve_project');
  for(const voice of ['drums','bass','vocals','other','piano']){
   await importStem(voice);await assertControls(false,true,true);
   assert.deepStrictEqual(await selected(),{guitar_engine:'general',allow_guitar_predicted_releases:false});
  }
  await analyze(general('provided'));await assertControls(false,false,false);
  checks.push('all_five_non_guitar_stems_exclude_specialist');
  await importStem('guitar');await main.check();await tail.check();
  await analyze(specialized('provided',true));await assertControls(true,true,true);
  assert.strictEqual(await page.locator('#pianoSpecialist').isChecked(),false);
  assert.strictEqual(await page.locator('#instrumentSpecialist').isChecked(),false);
  assert.strictEqual(await page.locator('#instrumentPredictedReleases').isChecked(),false);
  checks.push('guitar_does_not_enable_other_specialists');
  await page.locator('#guitarDetector').screenshot({path:path.join(output,'desktop.png')});
  await page.setViewportSize({width:390,height:844});
  overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
  await page.locator('#guitarDetector').screenshot({path:path.join(output,'mobile.png')});
  assert(!overflow,'Horizontal mobile overflow');checks.push('narrow_screen_no_horizontal_overflow');
  await page.route('**/api/status',async route=>{
   const response=await route.fetch(),json=await response.json();json.guitar.available=false;
   await route.fulfill({response,json});
  });
  await reopen(pid);await assertControls(false,true,true);
  assert.deepStrictEqual(await selected(),{guitar_engine:'general',allow_guitar_predicted_releases:false});
  assert((await page.locator('#guitarDetector').textContent()).includes('未包含'));
  await page.locator('#guitarDetector').screenshot({path:path.join(output,'missing-model.png')});
  checks.push('missing_model_disables_saved_choice');
  assert.deepStrictEqual(errors,[]);
  const report={passed:true,neural_inference:false,synthetic_backend:true,musical_acceptance:false,
   checks,submitted_options:requests,errors,overflow};
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));
 }catch(error){
  await page.screenshot({path:path.join(output,'failure.png'),fullPage:true}).catch(()=>{});
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({passed:false,neural_inference:false,
   synthetic_backend:true,musical_acceptance:false,checks,submitted_options:requests,errors,overflow,
   failure:String(error.stack||error)},null,2));
  throw error;
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1)});
