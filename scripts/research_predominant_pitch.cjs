'use strict';
// Optional, pinned Essentia.js research adapter. Does not ship the AGPL runtime.
// Input is original-level mono float32 little-endian PCM, already at 44100 Hz.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { performance } = require('node:perf_hooks');

const PINNED = {
  'essentia-wasm.umd.js': '7e0a2b5507199e8162c4ed090d38518de0c7faa070ce35d1593e7631b201014d',
  'essentia.js-core.umd.js': 'e3958a89ca0d3e1f95f67de627a383cdbbd8058f24fd53c6749758ff78b5e337',
};
const sha = value => crypto.createHash('sha256').update(value).digest('hex');

function run(runtime, inputPath, outputPath, expectedInputSha) {
  if (!runtime || !inputPath || !outputPath || !/^[a-f0-9]{64}$/.test(expectedInputSha || '')) {
    throw new Error('Usage: node research_predominant_pitch.cjs RUNTIME_DIST MONO_44100_F32 OUTPUT_JSON INPUT_SHA256');
  }
  if (fs.existsSync(outputPath)) throw new Error('Use a new output file; preserve earlier evidence');
  const bytes = fs.readFileSync(inputPath);
  if (sha(bytes) !== expectedInputSha) throw new Error('Input PCM identity differs');
  if (bytes.length % 4 || bytes.length < 44100 * 4 * .1 || bytes.length > 44100 * 4 * 1200) {
    throw new Error('Expected 0.1–1200 seconds of float32 mono PCM');
  }
  const pcm = new Float32Array(bytes.length / 4);
  for (let i = 0; i < pcm.length; ++i) pcm[i] = bytes.readFloatLE(i * 4);
  if (!pcm.every(Number.isFinite)) throw new Error('Nonfinite input PCM');
  for (const [name, digest] of Object.entries(PINNED)) {
    if (sha(fs.readFileSync(path.join(runtime, name))) !== digest) throw new Error('Runtime identity differs: ' + name);
  }
  const sampleRate = 44100, hop = 128, frame = 2048;
  const identity = {
    runner_sha256: sha(fs.readFileSync(__filename)), node: process.version,
    platform: process.platform, package_version: '0.1.3',
    package_git_head: 'f46c91c08bdf263d5f3d575ab8fb0f9b81695acf', runtime_files: PINNED,
    input_sha256: expectedInputSha, sample_rate: sampleRate, input_samples: pcm.length,
    duration_seconds: pcm.length / sampleRate, hop_samples: hop, frame_samples: frame,
    preprocessing: 'EqualLoudness at 44100 Hz; no gain normalization; no resampling in this adapter',
    output_clock: 'centered frames, first frame at input time zero; raw right-padding frames retained',
    confidence_semantics: 'algorithm pitch-contour confidence, not calibrated note presence or loudness',
    enabled_in_application: false, musical_acceptance: false,
  };
  let result;
  if (pcm.every(value => value === 0)) {
    // This pinned implementation throws on empty contours. Explicitly record
    // the bypass rather than claiming it returned an empty pitch track.
    result = { ...identity, status: 'exact_zero_pcm_skipped', algorithm_called: false,
      pitch_hz: null, pitch_confidence: null, elapsed_seconds: 0 };
  } else {
    const wasm = require(path.resolve(runtime, 'essentia-wasm.umd.js'));
    const Essentia = require(path.resolve(runtime, 'essentia.js-core.umd.js'));
    const engine = new Essentia(wasm);
    let vector, filtered, detected;
    const begin = performance.now();
    try {
      vector = engine.arrayToVector(pcm);
      filtered = engine.EqualLoudness(vector, sampleRate).signal;
      if (filtered.size() !== pcm.length) throw new Error('EqualLoudness changed the sample clock');
      detected = engine.PredominantPitchMelodia(filtered, 10, 3, frame, false, .8, hop,
        1, 40, 20000, 100, 80, 20, .9, .9, 27.5625, 55, sampleRate, 100, false, .2);
      const pitch = Array.from(engine.vectorToArray(detected.pitch));
      const confidence = Array.from(engine.vectorToArray(detected.pitchConfidence));
      if (pitch.length !== confidence.length || !pitch.every(Number.isFinite) || !confidence.every(Number.isFinite)) {
        throw new Error('Invalid pitch-contour output');
      }
      result = { ...identity, essentia_runtime_version: engine.version, status: 'ok', algorithm_called: true,
        pitch_hz: pitch, pitch_confidence: confidence, output_frames: pitch.length,
        frames_centered_beyond_input: pitch.filter((_, index) => index * hop >= pcm.length).length,
        elapsed_seconds: (performance.now() - begin) / 1000 };
    } catch (error) {
      result = { ...identity, essentia_runtime_version: engine.version, status: 'algorithm_failed',
        algorithm_called: true, error: String(error), pitch_hz: null, pitch_confidence: null,
        elapsed_seconds: (performance.now() - begin) / 1000 };
      process.exitCode = 1;
    } finally {
      if (detected) { detected.pitch.delete(); detected.pitchConfidence.delete(); }
      if (filtered) filtered.delete();
      if (vector) vector.delete();
      engine.shutdown(); engine.delete();
    }
  }
  if (sha(fs.readFileSync(inputPath)) !== expectedInputSha) throw new Error('Input file changed during analysis');
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n', { flag: 'wx' });
  console.log(JSON.stringify({status: result.status, output: outputPath, output_frames: result.output_frames,
    algorithm_called: result.algorithm_called, elapsed_seconds: result.elapsed_seconds}));
}

if (require.main === module) {
  try { run(...process.argv.slice(2)); }
  catch (error) { console.error(String(error)); process.exitCode = 1; }
}
