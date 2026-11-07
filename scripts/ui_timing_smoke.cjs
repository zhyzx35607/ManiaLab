const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:8976';await page.goto(base);
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/synthetic.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='synthetic');
 await page.locator('#analysisEngine').selectOption('basic');await page.locator('#analyze').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('采音候选'),{},{timeout:90000});
 await page.locator('#bpm').fill('120');await page.locator('#offset').fill('0');await page.locator('#timingConfirmed').check();
 const ratios=[];
 for(const ratio of ['1','0','0.6']){
  await page.locator('#ln_ratio').fill(ratio);await page.locator('#ln_ratio').dispatchEvent('input');
  await page.locator('#generate').click();await page.waitForFunction(()=>!document.querySelector('#generate').disabled,{},{timeout:60000});
  const pid=await page.locator('#recent').inputValue(),state=await(await page.request.get(base+'/api/projects/'+pid)).json();
  const v=state.variants[state.active];ratios.push(v.report.ln_ratio);
  if(Math.abs(v.report.ln_ratio-Number(ratio))>.001)throw Error(JSON.stringify(v.report));
 }
 await page.locator('#seek').fill('1');await page.locator('#seek').dispatchEvent('input');
 await page.screenshot({path:'tests/artifacts/alpha3-ln-controls.png',fullPage:true});
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/timing-137.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='timing-137');
 await page.locator('#analysisEngine').selectOption('basic');await page.locator('#analyze').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('采音候选'),{},{timeout:90000});
 const bpm=Number(await page.locator('#bpm').inputValue());if(Math.abs(bpm-137)>.01)throw Error('wrong BPM '+bpm);
 await page.locator('#checkTiming').click();await page.waitForFunction(()=>document.querySelector('#timingReport').textContent.includes('检查'));
 const check=await page.locator('#timingReport').textContent();
 await page.locator('#metronome').check();await page.locator('#audio').evaluate(a=>a.play());await page.waitForTimeout(500);await page.locator('#audio').evaluate(a=>a.pause());
 await page.locator('#halfBpm').click();if(Math.abs(Number(await page.locator('#bpm').inputValue())-bpm/2)>.001)throw Error('half BPM');
 await page.locator('#timingConfirmed').check();await page.locator('#offset').fill('300');
 if(await page.locator('#timingConfirmed').isChecked())throw Error('confirmation not reset');
 await page.locator('#tempoCandidates').selectOption('0');
 await page.locator('#generate').click();await page.waitForFunction(()=>!document.querySelector('#generate').disabled,{},{timeout:60000});
 if(await page.locator('#message').evaluate(e=>e.classList.contains('error')))throw Error(await page.locator('#message').textContent());
 await page.screenshot({path:'tests/artifacts/alpha3-timing-controls.png',fullPage:true});
 console.log(JSON.stringify({ratios,bpm,check,errors}));await browser.close();if(errors.length)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
