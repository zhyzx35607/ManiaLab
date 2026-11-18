const {test, expect} = require('@playwright/test');
const {execFileSync} = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');
const {pathToFileURL} = require('node:url');

const python = process.env.MANIALAB_TEST_PYTHON ||
  path.resolve('.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

const fixture = `
import hashlib, json, sys, wave
from pathlib import Path
source = Path(sys.argv[1])
source.mkdir(parents=True)
audio = source / 'silence.wav'
with wave.open(str(audio), 'wb') as out:
    out.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
    out.writeframes(b'\\x00\\x00' * 24000)
for model, pitch, t in [('kong', 60, 1010), ('transkun', 64, 2073)]:
    note = dict(voice='piano', pitch=pitch, start=t-10, end=t-10, audio_start=t)
    (source / f'{model}-10k-notes.json').write_text(json.dumps([note]), encoding='utf-8')
identity = dict(reference_chart_read=False, direct_listening_performed=False,
    source_sha256={'audio': hashlib.sha256(audio.read_bytes()).hexdigest()})
case = dict(name='transkun-10k', audio_reference=audio.name,
    delta_vs_kong={'added_by_voice': {'piano': 1}, 'removed_by_voice': {'piano': 1}})
(source / 'report.json').write_text(json.dumps(dict(identity=identity, cases=[case])), encoding='utf-8')
`;

const parseCSV = `
import csv, json, sys
with open(sys.argv[1], newline='', encoding='utf-8-sig') as source:
    print(json.dumps(list(csv.reader(source))))
`;

test('the offline piano review download matches the template and preserves ratings and comments', async ({page}, testInfo) => {
  const source = testInfo.outputPath('synthetic');
  const built = testInfo.outputPath('built-review');
  const output = testInfo.outputPath('refreshed-review');
  execFileSync(python, ['-c', fixture, source]);
  execFileSync(python, ['scripts/build_piano_onset_blind_review.py', '--input', source, '--output', built]);
  execFileSync(python, [
    'scripts/refresh_piano_blind_review.py',
    '--source-participant', path.join(built, 'participant'),
    '--output', output,
  ]);
  const participant = path.join(output, 'participant');
  const read = file => JSON.parse(execFileSync(python, ['-c', parseCSV, file], {encoding: 'utf8'}));
  const template = read(path.join(participant, 'review-template.csv'));
  expect(template[0]).toEqual(['clip_id', 'audio_file', 'audio_start_ms', 'pitch_midi', 'audibility', 'confidence_1_5', 'comment']);
  expect(template.slice(1).map(row => Number(row[2])).sort((a, b) => a-b)).toEqual([1010, 2073]);
  expect(await fs.readFile(path.join(built, 'participant/review-template.csv'))).toEqual(
    await fs.readFile(path.join(participant, 'review-template.csv')),
  );
  expect(await fs.readdir(output)).not.toContain('deblind-key.json');
  await page.goto(pathToFileURL(path.join(participant, 'index.html')).href);
  await expect(page.locator('tbody tr')).toHaveCount(2);
  const first = page.locator('tbody tr').first();
  await expect(first).toContainText((Number(template[1][2]) / 1000).toFixed(3) + 's');
  const comment = '听到 "钢琴", 还有回声\n下一拍较轻';
  await first.locator('[data-field="audibility"]').selectOption('yes');
  await first.locator('[data-field="confidence_1_5"]').selectOption('4');
  await first.locator('[data-field="comment"]').fill(comment);
  await page.reload();
  await expect(page.locator('#draft-status')).toContainText('恢复草稿');
  const restored = page.locator('tbody tr').first();
  await expect(restored.locator('[data-field="audibility"]')).toHaveValue('yes');
  await expect(restored.locator('[data-field="confidence_1_5"]')).toHaveValue('4');
  await expect(restored.locator('[data-field="comment"]')).toHaveValue(comment);
  const received = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await received;
  expect(download.suggestedFilename()).toBe('blind-review-results.csv');
  const file = await download.path();
  const bytes = await fs.readFile(file);
  expect(bytes.subarray(0, 3).toString('hex')).toBe('efbbbf');
  expect(read(file)).toEqual([template[0], [...template[1].slice(0, 4), 'yes', '4', comment], template[2]]);
  await page.locator('#clear-draft').click();
  await page.reload();
  await expect(page.locator('#draft-status')).not.toContainText('恢复草稿');
  await expect(page.locator('tbody tr').first().locator('[data-field="audibility"]')).toHaveValue('');
});
