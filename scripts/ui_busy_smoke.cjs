// A delayed request reproduces the interval where unsaved changes used to be
// silently overwritten by a finished analysis/generation response.
const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:8982';await page.goto(base);
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/synthetic.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='synthetic');
 await page.locator('[data-key="7"]').click();await page.locator('#density').fill('5');
 await page.locator('#analysisEngine').selectOption('basic');
 let release;const gate=new Promise(resolve=>release=resolve);
 await page.route('**/analyze',async route=>{await gate;await route.continue();});
 await page.locator('#analyze').click();
 const controls=['density','ln_ratio','mode','preset','bpm','offset','division','audioShift','waveAlign','auditionApply','analysisEngine'];
 for(const id of controls)if(!await page.locator('#'+id).isDisabled())throw Error('busy edit remains enabled: '+id);
 if(!await page.locator('[data-key="10"]').isDisabled())throw Error('busy keycount change remains enabled');
 if(await page.locator('#cancel').isDisabled())throw Error('cancel was disabled');
 release();await page.waitForFunction(()=>!document.querySelector('#analyze').disabled,{},{timeout:90000});
 for(const id of controls)if(await page.locator('#'+id).isDisabled())throw Error('finished edit remains disabled: '+id);
 if(await page.locator('#density').inputValue()!=='5'||!await page.locator('[data-key="7"]').evaluate(e=>e.classList.contains('active')))throw Error('original settings changed');
 await page.unroute('**/analyze');
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/public-maps/5900981.osu'));
 await page.waitForFunction(()=>document.querySelector('#projectSub').textContent.includes('1850'));
 if(!await page.locator('#bpm').isDisabled()||!await page.locator('#waveAlign').isDisabled())throw Error('strict-remap timing became editable');
 console.log(JSON.stringify({busyControls:controls.length+1,cancelAvailable:true,settingsPreserved:true,strictTimingProtected:true,errors}));
 await browser.close();if(errors.length)process.exit(1);
})().catch(error=>{console.error(error);process.exit(1)});
