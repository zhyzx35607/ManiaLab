const {test, expect} = require('@playwright/test');

function pcmCue(wave, time) {
  const rate = wave.readUInt32LE(24), stride = wave.readUInt16LE(32);
  const samples = wave.indexOf('data') + 8, at = Math.round(time * rate);
  const energy = hz => {
    let real = 0, imaginary = 0;
    for (let i = 0; i < Math.round(rate * .03); i++) {
      const value = wave.readInt16LE(samples + (at + i) * stride), phase = 2 * Math.PI * hz * i / rate;
      real += Math.cos(phase) * value;
      imaginary += Math.sin(phase) * value;
    }
    return real * real + imaginary * imaginary;
  };
  expect(wave.readInt16LE(samples + (at - 1) * stride)).toBe(0);
  expect(Math.abs(wave.readInt16LE(samples + (at + 1) * stride))).toBeGreaterThan(100);
  return energy(1600) > energy(1150);
}

async function playerWave(page) {
  const encoded = await page.locator('#auditionAudio').evaluate(player => {
    const blob = window.testAuditionBlobs.get(player.src);
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(blob);
    });
  });
  return Buffer.from(encoded, 'base64');
}

async function captureAuditions(page) {
  await page.goto('/');
  await page.evaluate(() => {
    const create = URL.createObjectURL.bind(URL);
    window.testAuditionBlobs = new Map();
    URL.createObjectURL = blob => {
      const url = create(blob);
      window.testAuditionBlobs.set(url, blob);
      return url;
    };
  });
}

test('imported live metronome and audition WAV agree across tempo and meter changes', async ({page}) => {
  await captureAuditions(page);
  await page.locator('#recent').selectOption({label: 'UI LN navigation remap'});
  await expect(page.locator('#mode')).toHaveValue('remap');
  await page.locator('.audition-panel').evaluate(panel => { panel.open = true; });
  await expect(page.locator('#auditionA')).toHaveText('试听原谱拍线');
  await expect(page.locator('#auditionB')).toBeDisabled();
  await expect(page.locator('#auditionOffset')).toBeDisabled();
  const beats = await page.evaluate(() => beatsBetween(0, 8));
  expect(beats.filter(beat => beat.accent).map(beat => Math.round(beat.time * 1000)))
    .toEqual([115, 1615, 3115, 4365, 5615, 6865]);
  expect(beats.filter(beat => Math.round(beat.time * 1000) === 3115)).toHaveLength(1);
  await page.locator('#auditionLength').selectOption('8');
  const received = page.waitForResponse('**/audition');
  await page.locator('#auditionA').click();
  const response = await received;
  expect(response.request().postDataJSON().beat_clock).toBe('original');
  expect(response.ok()).toBeTruthy();
  expect(Number(response.headers()['x-manialab-audition-events'])).toBe(beats.length);
  await expect(page.locator('#auditionNote')).toContainText('各段拍号');
  const wave = await playerWave(page);
  for (const beat of beats) expect(pcmCue(wave, beat.time)).toBe(beat.accent);

  await page.locator('#mode').selectOption('audio');
  await expect(page.locator('#auditionB')).toBeEnabled();
  await page.locator('#auditionOffset').fill('75');
  const compared = page.waitForResponse('**/audition');
  await page.locator('#auditionB').click();
  const fixed = await compared;
  expect(fixed.request().postDataJSON()).toMatchObject({beat_clock: 'fixed', offset: 75});
  expect(fixed.ok()).toBeTruthy();
  await expect(page.locator('#auditionNote')).toContainText('75 ms');
  expect(pcmCue(await playerWave(page), .075)).toBeTruthy();
  await page.locator('#auditionAudio').evaluate(player => player.pause());
});

test('audio draft, saved red line, live bars and A/B audition use the same meter and shift', async ({page}) => {
  await captureAuditions(page);
  await page.locator('#recent').selectOption({label: 'UI audio meter'});
  await expect(page.locator('#offset')).toHaveValue('93');
  const accents = () => page.evaluate(() =>
    beatsBetween(0, 8).filter(beat => beat.accent).map(beat => Math.round(beat.time * 1000)));
  const triple = [93, 1593, 3093, 4593, 6093, 7593];
  expect(await accents()).toEqual(triple);
  await page.locator('#generate').click();
  await expect(page.locator('#message')).toContainText('结构检查通过');
  const red = (await (await page.request.get(await page.locator('#exportOsu').getAttribute('href'))).text())
    .match(/\[TimingPoints\]\s*([^\r\n]+)/)[1].split(',').map(Number);
  expect(red.slice(0, 3)).toEqual([93, 500, 3]);
  expect(await accents()).toEqual(triple);
  await page.locator('.audition-panel').evaluate(panel => { panel.open = true; });
  await page.locator('#auditionLength').selectOption('8');
  for (const [button, offset, meter] of [['auditionA', 93, 3], ['auditionB', 192, 4]]) {
    if (button === 'auditionB') await page.locator('#auditionOffset').fill('192.5');
    const received = page.waitForResponse('**/audition');
    await page.locator('#' + button).click();
    const response = await received;
    expect(response.ok()).toBeTruthy();
    expect(response.request().postDataJSON().offset).toBe(offset);
    expect(Number(response.headers()['x-manialab-audition-meter'])).toBe(meter);
    await expect(page.locator('#auditionNote')).toContainText(`每小节 ${meter} 拍`);
    const wave = await playerWave(page);
    for (let index = 0; offset + index * 500 < 8000; index++) {
      expect(pcmCue(wave, (offset + index * 500) / 1000)).toBe(index % meter === 0);
    }
    await page.locator('#auditionAudio').evaluate(player => player.pause());
  }
  await page.locator('#auditionApply').click();
  await expect(page.locator('#offset')).toHaveValue('192');
  expect(await accents()).toEqual([192, 2192, 4192, 6192]);
  await page.locator('#offset').fill('192.5');
  await page.locator('#timingConfirmed').check();
  const sent = page.waitForRequest(request => request.url().endsWith('/generate'));
  await page.locator('#generate').click();
  expect((await sent).postDataJSON().offset).toBe(192);
  await expect(page.locator('#message')).toContainText('结构检查通过');
});
