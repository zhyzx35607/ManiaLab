const {chromium}=require('../test-data/ui/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1512,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 const base=process.env.MANIALAB_URL||'http://127.0.0.1:8976';await page.goto(base);
 async function analyze(){await page.locator('#analyze').click();await page.waitForFunction(()=>document.querySelector('#message').textContent.includes('采音候选'),{},{timeout:90000});}
 async function generate(){await page.locator('#generate').click();await page.waitForFunction(()=>!document.querySelector('#generate').disabled,{},{timeout:60000});}
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/synthetic.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='synthetic');await analyze();
 await generate();
 if(!(await page.locator('#message').textContent()).includes('Timing'))throw Error('unreliable timing was not blocked');
 await page.locator('#referenceTiming').fill('BPM: 210\nOffset: 91');await page.locator('#applyTiming').click();
 if(await page.locator('#offset').inputValue()!=='91'||await page.locator('#bpm').inputValue()!=='210')throw Error('reference not applied');
 await page.locator('#timingConfirmed').check();await page.locator('#ln_ratio').fill('1');await generate();
 if(await page.locator('#message').evaluate(e=>e.classList.contains('error')))throw Error(await page.locator('#message').textContent());
 const downloadPromise=page.waitForEvent('download');await page.locator('#exportOsu').click();const download=await downloadPromise;
 const out='tests/artifacts/alpha3-grid.osu';await download.saveAs(out);
 const text=fs.readFileSync(out,'utf8'),red=text.split('[TimingPoints]')[1].split('[HitObjects]')[0].trim().split('\n')[0].split(',').map(Number);
 if(red[0]!==91||Math.abs(red[1]-60000/210)>.000001)throw Error('wrong exported timing');
 let endpoints=0,maxError=0;
 for(const line of text.split('[HitObjects]')[1].trim().split('\n')){
   const p=line.split(','),times=[Number(p[2])];if(Number(p[3])&128)times.push(Number(p[5].split(':')[0]));
   for(const t of times){const error=Math.abs(t-(red[0]+Math.round((t-red[0])/(red[1]/4))*red[1]/4));maxError=Math.max(maxError,error);endpoints++;if(error>.501)throw Error('off grid '+line);}
 }
 await page.locator('#seek').fill('1');await page.locator('#seek').dispatchEvent('input');
 await page.screenshot({path:'tests/artifacts/alpha3-grid.png',fullPage:true});
 await page.locator('#offset').fill('118');
 if(await page.locator('#exportOsu').getAttribute('href'))throw Error('stale export enabled');
 await page.locator('#offset').fill('91');
 if(!await page.locator('#exportOsu').getAttribute('href'))throw Error('matching export disabled');
 await page.locator('#division').selectOption('3');await page.locator('#timingConfirmed').check();await generate();
 const triple=await(await page.request.get(base+await page.locator('#exportOsu').getAttribute('href'))).text();
 if(!triple.includes('BeatDivisor: 3'))throw Error('triplet editor grid mismatch');
 for(const line of triple.split('[HitObjects]')[1].trim().split('\n')){
   const p=line.split(','),times=[Number(p[2])];if(Number(p[3])&128)times.push(Number(p[5].split(':')[0]));
   for(const t of times)if(Math.abs(t-(91+Math.round((t-91)/(60000/210/3))*60000/210/3))>.501)throw Error('triplet off grid');
 }
 await analyze();
 if(await page.locator('#exportOsu').getAttribute('href'))throw Error('reanalysis kept stale candidate active');
 if(await page.locator('#variants').inputValue()!=='')throw Error('old candidate still selected');
 await page.locator('#files').setInputFiles(path.resolve('tests/artifacts/timing-137.wav'));
 await page.waitForFunction(()=>document.querySelector('#projectTitle').textContent==='timing-137');await analyze();
 if(await page.locator('#bpm').inputValue()!=='137')throw Error('BPM not normalized');
 await page.locator('#offset').fill('0');await page.locator('#refitTiming').click();
 await page.waitForFunction(()=>document.querySelector('#timingReport').textContent.includes('重新估计 Offset'));
 const refit=Number(await page.locator('#offset').inputValue());if(Math.abs(refit-347)>5||!Number.isInteger(refit))throw Error('refit failed');
 await page.setViewportSize({width:390,height:844});
 const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);
 const result={endpoints,maxError,refit,errors,mobileOverflow:overflow};
 console.log(JSON.stringify(result));await browser.close();if(errors.length||overflow)process.exit(1);
})().catch(e=>{console.error(e);process.exit(1)});
