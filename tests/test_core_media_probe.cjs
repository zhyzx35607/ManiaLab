'use strict';

const assert=require('node:assert/strict');
const fs=require('node:fs');
const os=require('node:os');
const path=require('node:path');
const test=require('node:test');
const playwright=require('@playwright/test');
const {FIXTURE_SCOPE,PACKAGED_SCOPE,assertAudioRangeEvidence,assertProjectRequestOrder,assertRapidSwitchPlayback,
  assertReport,packagedIdentity,parseArgs,
  main,recordExpectedProjectRequest,recordFirstProjectRequest,sha256,syntheticWav,waitForSignal}=
  require('../scripts/verify_core_media.cjs');

test('base URL is required and restricted to the local app root',()=>{
  assert.throws(()=>parseArgs([]),/--base-url is required/);
  assert.throws(()=>parseArgs(['--base-url','https://127.0.0.1:9000']),/HTTP loopback/);
  assert.throws(()=>parseArgs(['--base-url','http://example.test:9000']),/HTTP loopback/);
  assert.throws(()=>parseArgs(['--base-url','http://127.0.0.1:9000/subpath']),/app root/);
  assert.deepEqual(parseArgs(['--base-url','http://127.0.0.1:9000/']),{base_url:'http://127.0.0.1:9000'});
  assert.throws(()=>parseArgs(['--base-url','http://127.0.0.1:9000','--executable','ManiaLab.exe']),
    /must be supplied together/);
  assert.deepEqual(parseArgs(['--base-url','http://127.0.0.1:9000','--executable','ManiaLab.exe',
    '--verification-run-id','a'.repeat(32)]),{base_url:'http://127.0.0.1:9000',executable:'ManiaLab.exe',
    verification_run_id:'a'.repeat(32)});
  assert.throws(()=>parseArgs(['--base-url','http://127.0.0.1:9000','--executable','ManiaLab.exe',
    '--verification-run-id','invalid']),/32 lowercase hexadecimal/);
});

test('synthetic WAV contains valid mono PCM headers and distinct tones',()=>{
  const first=syntheticWav(440,2),second=syntheticWav(660,2);
  assert.equal(first.toString('ascii',0,4),'RIFF');
  assert.equal(first.toString('ascii',8,12),'WAVE');
  assert.equal(first.readUInt32LE(24),22050);
  assert.equal(first.readUInt16LE(22),1);
  assert.equal(first.readUInt16LE(34),16);
  assert.equal(first.readUInt32LE(40),first.length-44);
  assert.notEqual(sha256(first),sha256(second));
  assert.throws(()=>syntheticWav(20000),/Invalid synthetic WAV/);
});

test('missing expected audio requests reject after the configured timeout',async()=>{
  const pending=new Promise(()=>{});
  await assert.rejects(waitForSignal(pending,'synthetic request',5),/synthetic request timed out after 5 ms/);
});

test('expected audio request resolves before the timeout',async()=>{
  let resolve;
  const pending=new Promise(done=>{resolve=done;});
  const waiting=waitForSignal(pending,'synthetic request',1000);
  resolve('observed');
  assert.equal(await waiting,'observed');
});

test('probe failure remains primary when browser cleanup fails and report is written',async()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'manialab-core-media-close-'));
  const reportPath=path.join(directory,'report.json');
  const originalLaunch=playwright.chromium.launch;
  playwright.chromium.launch=async()=>({
    newPage:async()=>({request:{get:async()=>({ok:()=>false,status:()=>503})},on(){}}),
    close:async()=>{throw new Error('synthetic close failure');},
  });
  try{
    await assert.rejects(main(['--base-url','http://127.0.0.1:9000','--report',reportPath]),error=>{
      assert.match(error.message,/ManiaLab status check failed \(503\)/);
      assert.match(error.message,/Browser close failed: synthetic close failure/);
      assert.deepEqual(error.secondaryFailures,['Browser close failed: synthetic close failure']);
      return true;
    });
    const report=JSON.parse(fs.readFileSync(reportPath,'utf8'));
    assert.equal(report.status,'failed');
    assert.equal(report.failure,'ManiaLab status check failed (503)');
    assert.deepEqual(report.cleanup_failures,['Browser close failed: synthetic close failure']);
  }finally{
    playwright.chromium.launch=originalLaunch;
    fs.rmSync(directory,{recursive:true,force:true});
  }
});

test('successful probe is reported failed when browser cleanup fails',async()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'manialab-core-media-close-'));
  const reportPath=path.join(directory,'report.json');
  const report={schema:1,check:'core_media_session',status:'passed',run_id:'synthetic-run',
    base_url:'http://127.0.0.1:9000',scope:FIXTURE_SCOPE,
    steps:['synthetic_projects','audio_range','seek','play_pause','rapid_source_switch','browser_health']
      .map(name=>({name,status:'passed'}))};
  Object.assign(report.steps[4],{requested_projects:['project-b','project-a'],
    requested_project_ids:['project-b','project-a'],pre_switch_audio_project_ids:['project-a'],
    unexpected_audio_project_ids:[]});
  try{
    await assert.rejects(main(['--base-url','http://127.0.0.1:9000','--report',reportPath],{
      launchBrowser:async()=>({newPage:async()=>({}),close:async()=>{throw new Error('synthetic close failure');}}),
      probePage:async()=>report,
    }),/synthetic close failure/);
    const saved=JSON.parse(fs.readFileSync(reportPath,'utf8'));
    assert.equal(saved.status,'failed');
    assert.match(saved.failure,/Browser close failed: synthetic close failure/);
    assert.deepEqual(saved.cleanup_failures,['Browser close failed: synthetic close failure']);
  }finally{
    fs.rmSync(directory,{recursive:true,force:true});
  }
});

test('packaged identity hashes the executable in chunks and binds it to the manifest',()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'manialab-core-media-'));
  try{
    const executable=path.join(directory,'ManiaLab.exe');
    const bytes=Buffer.alloc(1024*1024+37,0x5a);
    fs.writeFileSync(executable,bytes);
    const manifest={build_profile:'core',capabilities:{audio:true},executable_sha256:sha256(bytes)};
    fs.writeFileSync(path.join(directory,'release-manifest.json'),JSON.stringify(manifest));
    const identity=packagedIdentity(executable,'c'.repeat(32));
    assert.deepEqual(identity,{build_profile:'core',capabilities:{audio:true},executable_sha256:sha256(bytes),
      release_manifest_sha256:sha256(Buffer.from(JSON.stringify(manifest))),verification_run_id:'c'.repeat(32)});
    fs.writeFileSync(path.join(directory,'release-manifest.json'),JSON.stringify({...manifest,executable_sha256:'0'.repeat(64)}));
    assert.throws(()=>packagedIdentity(executable,'c'.repeat(32)),/does not match its release manifest/);
  }finally{
    fs.rmSync(directory,{recursive:true,force:true});
  }
});

test('rapid audio requests must be the second project followed by the first',()=>{
  assert.deepEqual(assertProjectRequestOrder(['project-b','project-a'],['project-b','project-a']),
    ['project-b','project-a']);
  assert.throws(()=>assertProjectRequestOrder(['project-a','project-b'],['project-b','project-a']),
    /expected project order/);
  assert.throws(()=>assertProjectRequestOrder(['project-b','project-b'],['project-b','project-a']),
    /expected project order/);
  assert.throws(()=>assertProjectRequestOrder(['project-a','project-a'],['project-a','project-a']),
    /expected project order/);
});

test('audio Range evidence binds the exact response length and WAV byte count',()=>{
  assert.doesNotThrow(()=>assertAudioRangeEvidence(206,'bytes 0-63/264644',64,264644));
  for(const contentRange of ['bytes 0-63/264600','bytes 0-63/*','bytes 0-62/264644','bytes 0-63/264644 trailing'])
    assert.throws(()=>assertAudioRangeEvidence(206,contentRange,64,264644),/Audio range response is invalid/);
});

test('rapid source switch starts from zero and advances after switching back',()=>{
  assert.doesNotThrow(()=>assertRapidSwitchPlayback(1.5,0.1,0.35,6));
  assert.throws(()=>assertRapidSwitchPlayback(1.5,1.5,1.75,6),/Rapid project switch playback is invalid/);
  assert.throws(()=>assertRapidSwitchPlayback(1.3,0.1,0.35,6),/Rapid project switch playback is invalid/);
  assert.throws(()=>assertRapidSwitchPlayback(1.5,0.1,0.25,6),/Rapid project switch playback is invalid/);
});

test('repeated Range requests do not advance the distinct project order',()=>{
  const distinct=[];
  assert.equal(recordFirstProjectRequest(distinct,'project-b'),true);
  assert.equal(recordFirstProjectRequest(distinct,'project-b'),false);
  assert.equal(recordFirstProjectRequest(distinct,'project-a'),true);
  assert.equal(recordFirstProjectRequest(distinct,'project-a'),false);
  assert.deepEqual(distinct,['project-b','project-a']);
  assert.deepEqual(assertProjectRequestOrder(distinct,['project-b','project-a']),distinct);
});

test('pre-switch requests and repeated Range requests do not advance expected project order',()=>{
  const distinct=[];
  const expected=['project-b','project-a'];
  assert.equal(recordExpectedProjectRequest(distinct,'project-a','project-a',expected),false);
  assert.deepEqual(distinct,[]);
  assert.equal(recordExpectedProjectRequest(distinct,'project-b','project-b',expected),true);
  assert.equal(recordExpectedProjectRequest(distinct,'project-b','project-b',expected),false);
  assert.equal(recordExpectedProjectRequest(distinct,'project-a','project-b',expected),false);
  assert.deepEqual(distinct,['project-b']);
  assert.equal(recordExpectedProjectRequest(distinct,'project-a','project-a',expected),true);
  assert.deepEqual(distinct,['project-b','project-a']);
  assert.deepEqual(assertProjectRequestOrder(distinct,expected),distinct);
  assert.throws(()=>recordExpectedProjectRequest(
    [], 'project-a','project-a',['project-a','project-a']),
    /two distinct values/);
});

test('report validator requires six passed smoke steps and run identity',()=>{
  const report={schema:1,check:'core_media_session',status:'passed',run_id:'test-run',base_url:'http://127.0.0.1:9000',
    scope:FIXTURE_SCOPE,
    steps:['synthetic_projects','audio_range','seek','play_pause','rapid_source_switch','browser_health']
      .map(name=>({name,status:'passed'}))};
  report.steps[0].projects=[{id:'project-a',bytes:264644},{id:'project-b',bytes:264644}];
  Object.assign(report.steps[1],{http_status:206,content_range:'bytes 0-63/264644',bytes:64});
  Object.assign(report.steps[2],{target_seconds:1.25,actual_seconds:1.25,duration_seconds:6});
  report.steps[4].requested_projects=['project-b','project-a'];
  report.steps[4].requested_project_ids=['project-b','project-a'];
  report.steps[4].pre_switch_audio_project_ids=['project-a'];
  report.steps[4].unexpected_audio_project_ids=[];
  Object.assign(report.steps[4],{saved_seconds:1.5,return_start_seconds:0.1,final_seconds:0.35});
  assert.equal(assertReport(report),report);
  assert.throws(()=>assertReport({...report,steps:report.steps.slice(0,4)}),/incomplete/);
  assert.throws(()=>assertReport({...report,run_id:''}),/incomplete/);
  const reversed={...report,steps:report.steps.map(step=>({...step}))};
  reversed.steps[4].requested_project_ids=['project-a','project-b'];
  assert.throws(()=>assertReport(reversed),/incomplete/);
  const duplicate={...report,steps:report.steps.map(step=>({...step}))};
  duplicate.steps[4].requested_projects=['project-a','project-a'];
  duplicate.steps[4].requested_project_ids=['project-a','project-a'];
  assert.throws(()=>assertReport(duplicate),/incomplete/);
  const invalidPreSwitch={...report,steps:report.steps.map(step=>({...step}))};
  invalidPreSwitch.steps[4].pre_switch_audio_project_ids=['project-b'];
  assert.throws(()=>assertReport(invalidPreSwitch),/incomplete/);
  const unexpected={...report,steps:report.steps.map(step=>({...step}))};
  unexpected.steps[4].unexpected_audio_project_ids=['stale-project'];
  assert.throws(()=>assertReport(unexpected),/incomplete/);
  const restoredSavedPosition={...report,steps:report.steps.map(step=>({...step}))};
  restoredSavedPosition.steps[4].return_start_seconds=1.5;
  restoredSavedPosition.steps[4].final_seconds=1.75;
  assert.throws(()=>assertReport(restoredSavedPosition),/incomplete/);
  const packaged={...report,scope:PACKAGED_SCOPE,build_profile:'core',capabilities:{audio:true},
    executable_sha256:'a'.repeat(64),
    release_manifest_sha256:'b'.repeat(64),verification_run_id:'c'.repeat(32)};
  assert.equal(assertReport(packaged),packaged);
  assert.throws(()=>assertReport({...packaged,scope:FIXTURE_SCOPE}),/incomplete/);
  assert.throws(()=>assertReport({...packaged,executable_sha256:'invalid'}),/incomplete/);
});
