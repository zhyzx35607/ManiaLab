const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:8978';
 for(let attempt=0;attempt<40;attempt++){try{if((await page.request.get(base+'/api/status')).ok())break;}catch{}await new Promise(r=>setTimeout(r,500));}
 await page.goto(base);
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/synthetic.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='synthetic');
 await page.locator('#analyze').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('采音候选'),{},{timeout:60000});
 await page.locator('#generate').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('结构检查通过'),{},{timeout:30000});
 const baseline=await page.locator('#message').textContent();
 await page.locator('#seek').fill('1');await page.locator('#seek').dispatchEvent('input');
 await page.screenshot({path:'tests/artifacts/final-package.png',fullPage:true});
 const waitDownload=page.waitForEvent('download');await page.locator('#exportOsz').click();await(await waitDownload).saveAs('tests/artifacts/final-package.osz');
 // Exercise the original model through the packaged app, using the previously installed isolated engine.
 await page.locator('details').filter({has:page.locator('#engineGenerate')}).locator('summary').click();
 await page.locator('#engineSeconds').fill('8');await page.locator('#engineGenerate').click();
 await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('原始 mania 模型')||document.querySelector('#message').classList.contains('error'),{},{timeout:180000});
 const neural=await page.locator('#message').textContent();
 if(await page.locator('#message').evaluate(el=>el.classList.contains('error')))throw Error(neural);
 const pid=await page.locator('#recent').inputValue();
 const project=await(await page.request.get(base+'/api/projects/'+pid)).json();
 if(!project.variants[0].report.valid)throw Error('Neural output validation failed');
 const packed=await page.request.get(base+'/api/projects/'+pid+'/export/osz');if(!packed.ok())throw Error('Neural export failed');
 require('fs').writeFileSync('tests/artifacts/final-neural.osz',await packed.body());
 console.log(JSON.stringify({baseline,neural,errors,packaged:true}));
 await browser.close();if(errors.length)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
