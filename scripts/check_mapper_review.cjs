const {chromium}=require('../test-data/ui/node_modules/playwright');
const {pathToFileURL}=require('url');
const path=require('path');
(async()=>{
 const folder=path.resolve(process.argv[2]);
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 try{
  await page.goto(pathToFileURL(path.join(folder,'review.html')).href);
  await page.waitForFunction(()=>document.querySelector('audio').readyState>=1);
  const expected=await page.evaluate(()=>({charts:data.charts.length,duration:(data.end-data.start)/1000,tracks:data.charts.map(c=>c.name),sources:data.sources.map(s=>s.name)}));
  if(await page.locator('canvas').count()!==expected.charts)throw Error('Missing comparison');
  await page.locator('canvas').first().click({position:{x:300,y:60}});
  const playback=await page.evaluate(async()=>{const a=document.querySelector('audio');await a.play();return {duration:a.duration,time:a.currentTime,paused:a.paused}});
  if(playback.paused||playback.time<=0||Math.abs(playback.duration-expected.duration)>.1)throw Error('Shared audio clock did not play/seek');
  const track=expected.tracks.at(-1);
  await page.locator('#track').selectOption(track);
  await page.waitForFunction(()=>document.querySelector('audio').readyState>=2&&document.querySelector('audio').currentTime>1);
  const sources=await page.locator('#source option').count();
  if(sources>1){
   const before=await page.locator('audio').evaluate(a=>{a.pause();return a.currentTime});
   const selected=expected.sources.filter(s=>s!=='mix');
   for(const source of selected)await page.locator('#source').selectOption(source);
   const suffix='review-source-'+selected.at(-1)+'-'+track+'.wav';
   await page.waitForFunction(suffix=>document.querySelector('audio').readyState>=2&&document.querySelector('audio').src.endsWith(suffix),suffix);
   const after=await page.locator('audio').evaluate(a=>({time:a.currentTime,paused:a.paused}));
   if(!after.paused||Math.abs(after.time-before)>.1)throw Error('Source switch lost the shared clock');
  }
  await page.screenshot({path:path.join(folder,'review-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  await page.screenshot({path:path.join(folder,'review-mobile.png'),fullPage:true});
  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({ok:true,charts:expected.charts,sources,...playback,errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
