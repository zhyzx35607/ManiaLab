'use strict';

const {chromium}=require('@playwright/test');
const crypto=require('node:crypto');
const fs=require('node:fs');
const path=require('node:path');

const HELP=`Usage: node scripts/verify_core_media.cjs --base-url http://127.0.0.1:PORT [--report PATH]

Runs synthetic native-media checks against an already-running ManiaLab instance.
For packaged Core EXE verification, also pass --executable PATH and
--verification-run-id ID from the matching portable verification report.
The probe imports two short synthetic WAV projects into the instance data-dir.
`;

const PACKAGED_SCOPE='packaged Core EXE synthetic native-media smoke; not musical acceptance';
const FIXTURE_SCOPE='synthetic native-media smoke against a test fixture; not packaged-build or musical acceptance';

function parseArgs(argv){
  const values={};
  for(let index=0;index<argv.length;index++){
    const key=argv[index];
    if(key==='--help'||key==='-h')return {help:true};
    if(!['--base-url','--report','--executable','--verification-run-id'].includes(key))
      throw new Error(`Unknown argument: ${key}`);
    const value=argv[++index];
    if(!value||value.startsWith('--'))throw new Error(`${key} requires a value`);
    values[key.slice(2).replaceAll('-','_')]=value;
  }
  if(!values.base_url)throw new Error('--base-url is required');
  const url=new URL(values.base_url);
  if(url.protocol!=='http:'||!['127.0.0.1','localhost','[::1]'].includes(url.hostname))
    throw new Error('--base-url must be an HTTP loopback URL');
  if(url.pathname!=='/'||url.search||url.hash)throw new Error('--base-url must identify the app root');
  values.base_url=url.origin;
  if(Boolean(values.executable)!==Boolean(values.verification_run_id))
    throw new Error('--executable and --verification-run-id must be supplied together');
  if(values.verification_run_id&&!/^[0-9a-f]{32}$/.test(values.verification_run_id))
    throw new Error('--verification-run-id must be 32 lowercase hexadecimal characters');
  return values;
}

function syntheticWav(frequency,seconds=6,sampleRate=22050){
  if(!Number.isFinite(frequency)||frequency<=0||frequency>=sampleRate/2||
     !Number.isInteger(seconds)||seconds<2||seconds>20||
     !Number.isInteger(sampleRate)||sampleRate<8000||sampleRate>96000)
    throw new Error('Invalid synthetic WAV settings');
  const frameCount=seconds*sampleRate;
  const pcm=Buffer.alloc(frameCount*2);
  for(let frame=0;frame<frameCount;frame++){
    const fade=Math.min(1,frame/(sampleRate*.02),(frameCount-frame)/(sampleRate*.02));
    const value=Math.round(Math.sin(2*Math.PI*frequency*frame/sampleRate)*fade*7000);
    pcm.writeInt16LE(value,frame*2);
  }
  const header=Buffer.alloc(44);
  header.write('RIFF',0);header.writeUInt32LE(36+pcm.length,4);header.write('WAVE',8);
  header.write('fmt ',12);header.writeUInt32LE(16,16);header.writeUInt16LE(1,20);
  header.writeUInt16LE(1,22);header.writeUInt32LE(sampleRate,24);
  header.writeUInt32LE(sampleRate*2,28);header.writeUInt16LE(2,32);header.writeUInt16LE(16,34);
  header.write('data',36);header.writeUInt32LE(pcm.length,40);
  return Buffer.concat([header,pcm]);
}

function sha256(buffer){return crypto.createHash('sha256').update(buffer).digest('hex');}

function sha256File(filename){
  const descriptor=fs.openSync(filename,'r');
  const chunk=Buffer.allocUnsafe(1024*1024);
  const hash=crypto.createHash('sha256');
  try{
    let bytesRead;
    while((bytesRead=fs.readSync(descriptor,chunk,0,chunk.length,null))>0)
      hash.update(chunk.subarray(0,bytesRead));
  }finally{
    fs.closeSync(descriptor);
  }
  return hash.digest('hex');
}

function recordFirstProjectRequest(orderedProjectIds,projectId){
  if(!Array.isArray(orderedProjectIds)||typeof projectId!=='string'||!projectId)
    throw new Error('Audio request has an invalid project ID');
  if(orderedProjectIds.includes(projectId))return false;
  orderedProjectIds.push(projectId);
  return true;
}

function recordExpectedProjectRequest(orderedProjectIds,projectId,activeProjectId,expectedProjectIds){
  if(!Array.isArray(expectedProjectIds)||expectedProjectIds.length!==2||
     new Set(expectedProjectIds).size!==expectedProjectIds.length)
    throw new Error('Expected audio project IDs must be two distinct values');
  if(projectId!==activeProjectId)return false;
  if(projectId!==expectedProjectIds[orderedProjectIds.length])return false;
  return recordFirstProjectRequest(orderedProjectIds,projectId);
}

function assertProjectRequestOrder(actual,expected){
  if(!Array.isArray(actual)||!Array.isArray(expected)||actual.length!==expected.length||
     new Set(expected).size!==expected.length||actual.some((projectId,index)=>projectId!==expected[index]))
    throw new Error(`Audio requests did not follow the expected project order: ${JSON.stringify(actual)}`);
  return actual;
}

function assertAudioRangeEvidence(status,contentRange,bodyBytes,totalBytes){
  if(status!==206||contentRange!==`bytes 0-63/${totalBytes}`||bodyBytes!==64||
     !Number.isInteger(totalBytes)||totalBytes<=63)
    throw new Error(`Audio range response is invalid: ${status} ${contentRange}`);
}

function assertRapidSwitchPlayback(savedSeconds,returnStartSeconds,finalSeconds,durationSeconds){
  if(!Number.isFinite(savedSeconds)||savedSeconds<=1.3||
     !Number.isFinite(returnStartSeconds)||returnStartSeconds<0||returnStartSeconds>0.35||
     !Number.isFinite(finalSeconds)||!Number.isFinite(durationSeconds)||
     finalSeconds>=durationSeconds||finalSeconds-returnStartSeconds<=0.15)
    throw new Error(`Rapid project switch playback is invalid: saved=${savedSeconds}, start=${returnStartSeconds}, final=${finalSeconds}`);
}

function assertReport(report){
  const rapid=report?.steps?.find(step=>step.name==='rapid_source_switch');
  const requestOrder=rapid&&Array.isArray(rapid.requested_projects)&&rapid.requested_projects.length===2&&
    new Set(rapid.requested_projects).size===2&&
    Array.isArray(rapid.requested_project_ids)&&rapid.requested_project_ids.length===2&&
    rapid.requested_project_ids.every((projectId,index)=>projectId===rapid.requested_projects[index])&&
    Array.isArray(rapid.pre_switch_audio_project_ids)&&
    rapid.pre_switch_audio_project_ids.every(projectId=>projectId===rapid.requested_projects[1])&&
    Array.isArray(rapid.unexpected_audio_project_ids)&&rapid.unexpected_audio_project_ids.length===0;
  const packagedIdentity=report&&report.scope===PACKAGED_SCOPE&&report.build_profile==='core'&&
    report.capabilities&&typeof report.capabilities==='object'&&
    Object.values(report.capabilities).every(value=>typeof value==='boolean')&&
    /^[0-9a-f]{64}$/.test(report.executable_sha256||'')&&
    /^[0-9a-f]{64}$/.test(report.release_manifest_sha256||'')&&
    /^[0-9a-f]{32}$/.test(report.verification_run_id||'');
  const fixtureScope=report&&report.scope===FIXTURE_SCOPE&&report.build_profile===undefined&&
    report.executable_sha256===undefined&&report.release_manifest_sha256===undefined&&
    report.verification_run_id===undefined;
  if(!report||report.schema!==1||report.check!=='core_media_session'||
     typeof report.run_id!=='string'||!report.run_id||typeof report.base_url!=='string'||
     !(packagedIdentity||fixtureScope)||report.status!=='passed'||!Array.isArray(report.steps)||report.steps.length!==6||
     report.steps.some(step=>step.status!=='passed')||!requestOrder)
    throw new Error(`Core media verification report is incomplete: ${JSON.stringify(report)}`);
  const seek=report.steps.find(step=>step.name==='seek');
  const range=report.steps.find(step=>step.name==='audio_range');
  const projects=report.steps.find(step=>step.name==='synthetic_projects')?.projects;
  try{
    assertAudioRangeEvidence(range?.http_status,range?.content_range,range?.bytes,projects?.[0]?.bytes);
    assertRapidSwitchPlayback(rapid.saved_seconds,rapid.return_start_seconds,rapid.final_seconds,
      seek?.duration_seconds);
  }catch{
    throw new Error(`Core media verification report is incomplete: ${JSON.stringify(report)}`);
  }
  return report;
}

function deferred(){
  let resolve;
  const promise=new Promise(done=>{resolve=done;});
  return {promise,resolve};
}

function waitForSignal(promise,description,timeoutMs=10_000){
  if(!Number.isInteger(timeoutMs)||timeoutMs<1)throw new Error('Signal timeout must be a positive integer');
  let timer;
  const timeout=new Promise((_,reject)=>{
    timer=setTimeout(()=>reject(new Error(`${description} timed out after ${timeoutMs} ms`)),timeoutMs);
  });
  return Promise.race([promise,timeout]).finally(()=>clearTimeout(timer));
}

async function importSyntheticProject(page,baseUrl,token,title,frequency){
  const wav=syntheticWav(frequency);
  const response=await page.request.post(`${baseUrl}/api/import`,{
    headers:{'X-ManiaLab-Token':token},
    multipart:{files:{name:`${title}.wav`,mimeType:'audio/wav',buffer:wav}},
  });
  if(!response.ok())throw new Error(`Synthetic project import failed (${response.status()}): ${await response.text()}`);
  const project=await response.json();
  return {id:project.id,title:project.title,filename:project.audio,bytes:wav.length,sha256:sha256(wav)};
}

async function selectProject(page,id){
  await page.locator('#recent').selectOption(id);
  await page.waitForFunction(projectId=>state?.id===projectId, id);
  await page.waitForFunction(projectId=>{
    const player=document.querySelector('#audio');
    return player&&player.src.includes(`/api/projects/${projectId}/audio`)&&
      player.readyState>=player.HAVE_METADATA&&Number.isFinite(player.duration);
  },id);
}

async function probePage(page,baseUrl,runId=crypto.randomUUID(),identity=null){
  const report={schema:1,check:'core_media_session',status:'running',run_id:runId,base_url:baseUrl,
    scope:identity?PACKAGED_SCOPE:FIXTURE_SCOPE,...identity,steps:[]};
  const add=(name,details)=>report.steps.push({name,status:'passed',...details});
  try{
  const errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  const status=await page.request.get(`${baseUrl}/api/status`);
  if(!status.ok())throw new Error(`ManiaLab status check failed (${status.status()})`);
  const statusJson=await status.json();
  if(typeof statusJson.version!=='string')throw new Error('ManiaLab status response has no version');
  await page.goto(baseUrl);
  const token=await page.locator('meta[name="manialab-token"]').getAttribute('content');
  if(!token)throw new Error('ManiaLab page has no session token');

  const suffix=runId.replaceAll('-','').slice(0,12);
  const first=await importSyntheticProject(page,baseUrl,token,`core-media-a-${suffix}`,440);
  const second=await importSyntheticProject(page,baseUrl,token,`core-media-b-${suffix}`,660);
  if(first.id===second.id)throw new Error('Synthetic media imports returned the same project ID');
  add('synthetic_projects',{projects:[first,second]});

  const range=await page.request.get(`${baseUrl}/api/projects/${first.id}/audio?audio=${encodeURIComponent(first.filename)}`,
    {headers:{Range:'bytes=0-63'}});
  const contentRange=range.headers()['content-range']||'';
  assertAudioRangeEvidence(range.status(),contentRange,(await range.body()).length,first.bytes);
  add('audio_range',{http_status:range.status(),content_range:contentRange,bytes:64});

  await page.reload();
  await selectProject(page,first.id);
  const player=page.locator('#audio');
  await player.evaluate(element=>{element.muted=true;element.currentTime=1.25;});
  await page.waitForFunction(()=>{
    const element=document.querySelector('#audio');
    return !element.seeking&&Math.abs(element.currentTime-1.25)<.1;
  });
  const seekTime=await player.evaluate(element=>element.currentTime);
  add('seek',{target_seconds:1.25,actual_seconds:seekTime,duration_seconds:await player.evaluate(element=>element.duration)});

  await player.evaluate(element=>element.play());
  await page.waitForFunction(start=>!audio.paused&&audio.currentTime>start+.15,seekTime);
  await player.evaluate(element=>element.pause());
  await page.waitForFunction(()=>audio.paused);
  const pausedAt=await player.evaluate(element=>element.currentTime);
  await page.waitForTimeout(200);
  const afterPause=await player.evaluate(element=>element.currentTime);
  if(!await player.evaluate(element=>element.paused)||Math.abs(afterPause-pausedAt)>.04)
    throw new Error('Native audio position did not remain stable after pause');
  add('play_pause',{started_at_seconds:seekTime,paused_at_seconds:pausedAt,after_pause_seconds:afterPause});

  await player.evaluate(element=>element.play());
  await page.waitForFunction(()=>!audio.paused);
  const savedTime=await player.evaluate(element=>element.currentTime);
  const gate=deferred(),firstRequest=deferred(),secondRequest=deferred();
  const held=[];
  const requestedProjectIds=[];
  const expectedProjectIds=[second.id,first.id];
  const preSwitchProjectIds=[];
  const unexpectedProjectIds=[];
  let returnStartTime;
  const handler=async route=>{
    const url=new URL(route.request().url());
    const match=url.pathname.match(/^\/api\/projects\/([^/]+)\/audio$/);
    if(!match)return route.continue();
    const projectId=decodeURIComponent(match[1]);
    const activeProjectId=await page.evaluate(()=>state?.id||null);
    const firstSeen=recordExpectedProjectRequest(
      requestedProjectIds,projectId,activeProjectId,expectedProjectIds
    );
    if(!firstSeen){
      if(requestedProjectIds.includes(projectId)){
        held.push(route);
        await gate.promise;
        try{await route.continue();}catch{}
        return;
      }
      if(requestedProjectIds.length===0&&projectId===activeProjectId&&
         projectId===expectedProjectIds[1]){
        preSwitchProjectIds.push(projectId);
        return route.continue();
      }
      unexpectedProjectIds.push(projectId);
      return route.continue();
    }
    held.push(route);
    if(firstSeen&&requestedProjectIds.length===1)firstRequest.resolve();
    if(firstSeen&&requestedProjectIds.length===2)secondRequest.resolve();
    await gate.promise;
    try{await route.continue();}catch{}
  };
  await page.route(url=>url.pathname.startsWith('/api/projects/')&&url.pathname.endsWith('/audio'),handler);
  try{
    await page.locator('#recent').selectOption(second.id);
    await page.waitForFunction(projectId=>state?.id===projectId,second.id);
    await waitForSignal(firstRequest.promise,'Second-project audio request was not observed');
    await page.locator('#recent').selectOption(first.id);
    await page.waitForFunction(projectId=>state?.id===projectId,first.id);
    await waitForSignal(secondRequest.promise,'Return-project audio request was not observed');
    await page.waitForFunction(projectId=>{
      const element=document.querySelector('#audio');
      return state?.id===projectId&&element&&element.src.includes(`/api/projects/${projectId}/audio`)&&
        Number.isFinite(element.currentTime);
    },first.id);
    returnStartTime=await player.evaluate(element=>element.currentTime);
  }finally{
    gate.resolve();
  }
  assertProjectRequestOrder(requestedProjectIds,expectedProjectIds);
  if(unexpectedProjectIds.length)throw new Error(`Unexpected audio project requests: ${JSON.stringify(unexpectedProjectIds)}`);
  await page.unroute(url=>url.pathname.startsWith('/api/projects/')&&url.pathname.endsWith('/audio'),handler);
  await page.waitForFunction(({projectId,start})=>{
    const element=document.querySelector('#audio');
    return state?.id===projectId&&element&&element.src.includes(`/api/projects/${projectId}/audio`)&&
      element.readyState>=element.HAVE_METADATA&&!element.paused&&element.currentTime>start+.15;
  },{projectId:first.id,start:returnStartTime});
  const finalTime=await player.evaluate(element=>element.currentTime);
  const duration=await player.evaluate(element=>element.duration);
  assertRapidSwitchPlayback(savedTime,returnStartTime,finalTime,duration);
  add('rapid_source_switch',{requested_projects:expectedProjectIds,
    requested_project_ids:requestedProjectIds,final_project:first.id,
    held_audio_requests:held.length,pre_switch_audio_project_ids:preSwitchProjectIds,
    unexpected_audio_project_ids:unexpectedProjectIds,saved_seconds:savedTime,
    return_start_seconds:returnStartTime,final_seconds:finalTime});
  await player.evaluate(element=>element.pause());
  await page.waitForFunction(()=>audio.paused);
  if(errors.length)throw new Error(`Browser page errors: ${errors.join('; ')}`);
  add('browser_health',{page_errors:errors.length,version:statusJson.version});
  report.status='passed';
  return assertReport(report);
  }catch(error){
    report.status='failed';
    report.failure=error.message||String(error);
    error.verificationReport=report;
    throw error;
  }
}

async function main(argv=process.argv.slice(2),dependencies={}){
  const options=parseArgs(argv);
  if(options.help){process.stdout.write(HELP);return 0;}
  const identity=options.executable?packagedIdentity(options.executable,options.verification_run_id):null;
  const launchBrowser=dependencies.launchBrowser||(()=>chromium.launch({headless:true}));
  const runProbe=dependencies.probePage||probePage;
  const browser=await launchBrowser();
  let report;let failure;const cleanupFailures=[];
  try{
    const page=await browser.newPage({viewport:{width:1280,height:900}});
    report=await runProbe(page,options.base_url,crypto.randomUUID(),identity);
  }catch(error){failure=error;report=error.verificationReport;}
  try{await browser.close();}catch(error){
    const diagnostic=`Browser close failed: ${error.message||String(error)}`;
    cleanupFailures.push(diagnostic);
    if(failure){
      failure.secondaryFailures=[...(failure.secondaryFailures||[]),diagnostic];
      const originalStack=failure.stack;
      failure.message=`${failure.message||String(failure)}; secondary cleanup failure: ${diagnostic}`;
      if(originalStack){
        const stackLineEnd=originalStack.indexOf('\n');
        failure.stack=`${failure.name||'Error'}: ${failure.message}${stackLineEnd<0?'':originalStack.slice(stackLineEnd)}`;
      }
    }else failure=error;
  }
  if(report){
    if(cleanupFailures.length){
      report.cleanup_failures=cleanupFailures;
      if(report.status==='passed'){
        report.status='failed';
        report.failure=`Cleanup failed: ${cleanupFailures.join('; ')}`;
      }
    }
    if(report.status==='passed')assertReport(report);
    const json=JSON.stringify(report,null,2)+'\n';
    if(options.report){
      const target=path.resolve(options.report);
      fs.mkdirSync(path.dirname(target),{recursive:true});
      fs.writeFileSync(target,json,{encoding:'utf8',flag:'wx'});
    }
    process.stdout.write(json);
  }
  if(failure)throw failure;
  return 0;
}

function packagedIdentity(executable,verificationRunId){
  const executablePath=path.resolve(executable);
  const manifestPath=path.join(path.dirname(executablePath),'release-manifest.json');
  const executableSha=sha256File(executablePath);
  const manifestBytes=fs.readFileSync(manifestPath);
  const manifest=JSON.parse(manifestBytes.toString('utf8'));
  if(manifest.build_profile!=='core')throw new Error('Core media verification requires a Core EXE');
  if(manifest.executable_sha256!==executableSha)
    throw new Error('Core media EXE hash does not match its release manifest');
  if(!/^[0-9a-f]{32}$/.test(verificationRunId))
    throw new Error('Core media verification run ID is invalid');
  return {build_profile:'core',capabilities:manifest.capabilities,executable_sha256:executableSha,
    release_manifest_sha256:sha256(manifestBytes),verification_run_id:verificationRunId};
}

if(require.main===module){
  main().catch(error=>{process.stderr.write(`${error.stack||error}\n`);process.exitCode=1;});
}

module.exports={FIXTURE_SCOPE,HELP,PACKAGED_SCOPE,assertAudioRangeEvidence,assertProjectRequestOrder,
  assertRapidSwitchPlayback,assertReport,main,packagedIdentity,parseArgs,probePage,
  recordExpectedProjectRequest,recordFirstProjectRequest,sha256,syntheticWav,waitForSignal};
