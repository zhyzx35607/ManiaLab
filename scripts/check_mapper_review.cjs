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
  if(await page.locator('canvas').count()!==3)throw Error('Missing comparison');
  await page.locator('canvas').first().click({position:{x:300,y:60}});
  const playback=await page.evaluate(async()=>{const a=document.querySelector('audio');await a.play();return {duration:a.duration,time:a.currentTime,paused:a.paused}});
  if(playback.paused||playback.time<=0||Math.abs(playback.duration-12)>.1)throw Error('Shared audio clock did not play/seek');
  await page.locator('#track').selectOption('review-adapter.wav');
  await page.waitForFunction(()=>document.querySelector('audio').readyState>=2&&document.querySelector('audio').currentTime>1);
  await page.screenshot({path:path.join(folder,'review-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  await page.screenshot({path:path.join(folder,'review-mobile.png'),fullPage:true});
  if(errors.length)throw Error(errors.join('\n'));
  console.log(JSON.stringify({ok:true,charts:3,...playback,errors}));
 }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});
