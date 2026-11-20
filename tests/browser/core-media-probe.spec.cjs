const {test,expect}=require('@playwright/test');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs');
const path=require('node:path');

const ROOT=path.resolve(__dirname,'../..');
const HELPER_TIMEOUT_MS=90_000;
const TEST_TIMEOUT_MS=120_000;

test('standalone Core media probe passes against the disposable local Flask fixture',async({},testInfo)=>{
  testInfo.setTimeout(TEST_TIMEOUT_MS);
  expect(testInfo.timeout).toBeGreaterThan(HELPER_TIMEOUT_MS);
  const baseUrl=testInfo.project.use.baseURL;
  expect(baseUrl).toMatch(/^http:\/\/127\.0\.0\.1:\d+$/);
  const reportPath=testInfo.outputPath('core-media-report.json');
  const stdout=execFileSync(process.execPath,[
    path.join(ROOT,'scripts/verify_core_media.cjs'),'--base-url',baseUrl,'--report',reportPath,
  ],{cwd:ROOT,encoding:'utf8',timeout:HELPER_TIMEOUT_MS,windowsHide:true});
  const report=JSON.parse(fs.readFileSync(reportPath,'utf8'));
  expect(report.base_url).toBe(baseUrl);
  expect(report.status).toBe('passed');
  expect(report.run_id).toMatch(/^[0-9a-f-]{36}$/);
  expect(report.steps.map(step=>step.name)).toEqual([
    'synthetic_projects','audio_range','seek','play_pause','rapid_source_switch','browser_health',
  ]);
  expect(report.steps.every(step=>step.status==='passed')).toBe(true);
  const projects=report.steps.find(step=>step.name==='synthetic_projects').projects;
  expect(projects[0].id).not.toBe(projects[1].id);
  const rapid=report.steps.find(step=>step.name==='rapid_source_switch');
  expect(rapid.requested_project_ids).toEqual([projects[1].id,projects[0].id]);
  expect(Array.isArray(rapid.pre_switch_audio_project_ids)).toBe(true);
  expect(rapid.pre_switch_audio_project_ids.every(projectId=>projectId===projects[0].id)).toBe(true);
  expect(rapid.unexpected_audio_project_ids).toEqual([]);
  expect(rapid.held_audio_requests).toBeGreaterThanOrEqual(rapid.requested_project_ids.length);
  expect(rapid.saved_seconds).toBeGreaterThan(1.3);
  expect(rapid.return_start_seconds).toBeGreaterThanOrEqual(0);
  expect(rapid.return_start_seconds).toBeLessThanOrEqual(0.35);
  expect(rapid.final_seconds-rapid.return_start_seconds).toBeGreaterThan(0.15);
  expect(rapid.final_seconds).toBeLessThan(report.steps.find(step=>step.name==='seek').duration_seconds);
  expect(report.steps.find(step=>step.name==='audio_range').content_range)
    .toBe(`bytes 0-63/${projects[0].bytes}`);
  expect(report.scope).toContain('test fixture');
  expect(report.scope).toContain('not packaged-build');
  expect(report.scope).toContain('musical acceptance');
  await testInfo.attach('core-media-report',{body:JSON.stringify(report,null,2),contentType:'application/json'});
  expect(stdout).toContain(report.run_id);
});
