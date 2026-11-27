(function (global) {
  'use strict';

  const MAX_DPR = 2;
  const LANE_COLORS = [
    '#168b73', '#327ab7', '#bb6737', '#755eaa', '#168b73',
    '#327ab7', '#bb6737', '#755eaa', '#168b73', '#327ab7',
  ];

  function formatTime(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const remainder = (value % 60).toFixed(1).padStart(4, '0');
    return `${String(minutes).padStart(2, '0')}:${remainder}`;
  }

  function finite(value, fallback = 0) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function normalizeNotes(notes, keys) {
    if (!Array.isArray(notes)) return [];
    const result = [];
    for (const note of notes) {
      if (!note || typeof note !== 'object') continue;
      const start = finite(note.start_ms ?? note.start ?? note.time_ms ?? note.time, NaN);
      const end = finite(note.end_ms ?? note.end ?? note.endTime, start);
      const lane = finite(note.lane ?? note.column ?? note.track, NaN);
      if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isInteger(lane)) continue;
      if (start < 0 || end < start || lane < 0 || lane >= keys) continue;
      result.push({
        start,
        end: Math.max(start, end),
        lane,
        hold: Boolean(note.is_hold ?? note.isHold ?? note.hold) || end > start,
      });
    }
    result.sort((a, b) => a.start - b.start || a.lane - b.lane || a.end - b.end);
    return result;
  }

  function peakHeld(notes) {
    const events = [];
    for (const note of notes) {
      if (!note.hold) continue;
      events.push([note.start, 1], [note.end, -1]);
    }
    events.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let held = 0;
    let maximum = 0;
    for (const [, delta] of events) {
      held += delta;
      maximum = Math.max(maximum, held);
    }
    return maximum;
  }

  function buildStats(notes, supplied) {
    const input = supplied && typeof supplied === 'object' ? supplied : {};
    const holds = notes.filter(note => note.hold).length;
    const objectCount = finite(input.objects ?? input.object_count, notes.length);
    const ratio = Number.isFinite(Number(input.ln_ratio ?? input.hold_note_ratio))
      ? Number(input.ln_ratio ?? input.hold_note_ratio)
      : (notes.length ? holds / notes.length : 0);
    const held = finite(input.peak_held ?? input.peakHeld, peakHeld(notes));
    return [
      { label: '物件', value: Math.round(objectCount).toLocaleString() },
      { label: 'LN 占比', value: `${(ratio * 100).toFixed(1)}%` },
      { label: '峰值同时 LN', value: Math.round(held).toString() },
      { label: '长条数量', value: Math.round(finite(input.hold_notes ?? input.ln_count, holds)).toString() },
    ];
  }

  function torosuSummary(stats) {
    const torosu = stats?.torosu && typeof stats.torosu === 'object' ? stats.torosu : stats || {};
    const values = [];
    for (const [label, keys] of [
      ['Minijack', ['minijack', 'minijack_count']],
      ['Pill', ['pill', 'pill_count']],
      ['手均衡', ['hand_balance', 'handBalance']],
      ['小拇指均衡', ['pinky_balance', 'outer_lane_balance', 'pinkyBalance']],
    ]) {
      const key = keys.find(name => torosu[name] !== undefined && torosu[name] !== null);
      if (key) {
        const value = torosu[key];
        values.push(`${label} ${typeof value === 'number' && label.includes('均衡') ? value.toFixed(2) : value}`);
      }
    }
    const distribution = torosu.lane_distribution ?? torosu.track_counts ?? torosu.lanes;
    if (Array.isArray(distribution) && distribution.length) {
      values.push(`轨道分布 ${distribution.map(value => Math.round(finite(value))).join(' / ')}`);
    } else if (distribution && typeof distribution === 'object') {
      const entries = Object.entries(distribution)
        .sort(([a], [b]) => finite(a) - finite(b))
        .map(([, value]) => Math.round(finite(value)));
      if (entries.length) values.push(`轨道分布 ${entries.join(' / ')}`);
    }
    return values.length ? values.join('　·　') : 'Torosu 键型统计尚未提供';
  }

  function mount(root, audio) {
    if (!(root instanceof Element)) throw new TypeError('Preview root must be an Element.');
    if (!audio || typeof audio.addEventListener !== 'function') {
      throw new TypeError('Preview audio must be an HTMLMediaElement.');
    }
    const host = document.createElement('section');
    host.className = 'ml-preview';
    host.innerHTML = `
      <div class="ml-preview-toolbar">
        <button class="ml-preview-play" type="button" data-preview-play aria-label="播放预览" title="播放预览">▶</button>
        <input class="ml-preview-progress" type="range" data-preview-progress min="0" max="0" step="0.01" value="0" aria-label="预览进度" />
        <output class="ml-preview-time" data-preview-time aria-live="off">00:00.0</output>
      </div>
      <div class="ml-preview-controls">
        <label class="ml-preview-control"><span>下落速度</span><output data-preview-scroll-value>0.8x</output><input type="range" data-preview-scroll min="0.5" max="1.5" step="0.1" value="0.8" aria-label="下落速度" /></label>
        <label class="ml-preview-control"><span>播放倍速</span><output data-preview-rate-value>1.0x</output><input type="range" data-preview-rate min="0.5" max="1.5" step="0.1" value="1" aria-label="播放倍速" /></label>
      </div>
      <div class="ml-preview-stage">
        <canvas class="ml-preview-canvas" data-preview-canvas aria-label="竖屏下落式谱面预览"></canvas>
        <div class="ml-preview-empty" data-preview-empty>选择已生成的谱面后显示实时预览</div>
      </div>
      <div class="ml-preview-stats" data-preview-stats aria-label="谱面健康摘要"></div>
      <p class="ml-preview-torosu" data-preview-torosu>Torosu 键型统计尚未提供</p>
    `;
    root.replaceChildren(host);
    const canvas = host.querySelector('[data-preview-canvas]');
    const context = canvas.getContext('2d');
    const playButton = host.querySelector('[data-preview-play]');
    const progress = host.querySelector('[data-preview-progress]');
    const timeOutput = host.querySelector('[data-preview-time]');
    const scrollInput = host.querySelector('[data-preview-scroll]');
    const scrollOutput = host.querySelector('[data-preview-scroll-value]');
    const rateInput = host.querySelector('[data-preview-rate]');
    const rateOutput = host.querySelector('[data-preview-rate-value]');
    const empty = host.querySelector('[data-preview-empty]');
    const statsHost = host.querySelector('[data-preview-stats]');
    const torosuHost = host.querySelector('[data-preview-torosu]');
    const controller = new AbortController();
    const { signal } = controller;
    let chart = { notes: [], keys: 10, bpm: 120, offset: 0, division: 4, meter: 4, stats: {} };
    let raf = 0;
    let observer = null;
    let destroyed = false;
    let width = 0;
    let height = 0;

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(MAX_DPR, Math.max(1, global.devicePixelRatio || 1));
      width = rect.width;
      height = rect.height;
      const targetWidth = Math.max(1, Math.round(width * dpr));
      const targetHeight = Math.max(1, Math.round(height * dpr));
      if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
        canvas.width = targetWidth;
        canvas.height = targetHeight;
      }
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw();
    }

    function updateProgress() {
      const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
      progress.max = String(Math.max(duration, 0));
      progress.value = String(Math.min(finite(audio.currentTime), duration || finite(audio.currentTime)));
      timeOutput.value = `${formatTime(audio.currentTime)} / ${formatTime(duration)}`;
      timeOutput.textContent = timeOutput.value;
      const playing = !audio.paused && !audio.ended;
      playButton.textContent = playing ? 'Ⅱ' : '▶';
      playButton.setAttribute('aria-label', playing ? '暂停预览' : '播放预览');
      playButton.title = playing ? '暂停预览' : '播放预览';
    }

    function renderStats() {
      statsHost.replaceChildren();
      for (const item of buildStats(chart.notes, chart.stats)) {
        const cell = document.createElement('div');
        cell.className = 'ml-preview-stat';
        const strong = document.createElement('strong');
        strong.textContent = item.value;
        const label = document.createElement('small');
        label.textContent = item.label;
        cell.append(strong, label);
        statsHost.append(cell);
      }
      torosuHost.textContent = torosuSummary(chart.stats);
    }

    function draw() {
      if (destroyed || !context || !width || !height) return;
      context.clearRect(0, 0, width, height);
      context.fillStyle = '#101a28';
      context.fillRect(0, 0, width, height);
      const current = Math.max(0, finite(audio.currentTime));
      const keys = chart.keys;
      const side = Math.max(24, Math.min(42, width * 0.09));
      const top = 24;
      const bottom = height - Math.max(34, height * 0.075);
      const laneWidth = (width - side * 2) / keys;
      const lineHeight = bottom - top;
      const pixelsPerSecond = 70 * finite(scrollInput.value, 0.8);
      const beatSeconds = 60 / chart.bpm;
      const stepSeconds = beatSeconds / Math.max(1, chart.division);
      const horizon = lineHeight / pixelsPerSecond;
      const lowerTime = Math.max(0, current - 0.28);
      const upperTime = current + horizon;

      for (let lane = 0; lane < keys; lane += 1) {
        const x = side + lane * laneWidth;
        context.fillStyle = lane % 2 === 0 ? '#152332' : '#121f2c';
        context.fillRect(x, top, laneWidth, height - top);
        context.fillStyle = LANE_COLORS[lane % LANE_COLORS.length];
        context.globalAlpha = 0.78;
        context.fillRect(x, top, laneWidth, 3);
        context.globalAlpha = 1;
        context.strokeStyle = '#34475b';
        context.lineWidth = 0.75;
        context.beginPath();
        context.moveTo(x, top);
        context.lineTo(x, height);
        context.stroke();
        context.fillStyle = '#c3d0dd';
        context.font = '11px "Segoe UI","Microsoft YaHei",sans-serif';
        context.textAlign = 'center';
        context.fillText(String(lane + 1), x + laneWidth / 2, height - 12);
      }
      context.strokeStyle = '#34475b';
      context.beginPath();
      context.moveTo(side + keys * laneWidth, top);
      context.lineTo(side + keys * laneWidth, height);
      context.stroke();

      if (stepSeconds > 0 && Number.isFinite(stepSeconds)) {
        const firstStep = Math.floor((lowerTime - chart.offset / 1000) / stepSeconds);
        const lastStep = Math.ceil((upperTime - chart.offset / 1000) / stepSeconds);
        for (let step = firstStep; step <= lastStep; step += 1) {
          const time = chart.offset / 1000 + step * stepSeconds;
          if (time < 0 || time < lowerTime || time > upperTime) continue;
          const y = bottom - (time - current) * pixelsPerSecond;
          const isBeat = step % chart.division === 0;
          const isBar = isBeat && Math.round(step / chart.division) % chart.meter === 0;
          context.strokeStyle = isBar ? '#72859a' : isBeat ? '#4b6178' : '#28394c';
          context.lineWidth = isBar ? 1.5 : isBeat ? 1 : 0.6;
          context.beginPath();
          context.moveTo(side, y);
          context.lineTo(side + keys * laneWidth, y);
          context.stroke();
          if (isBeat) {
            context.fillStyle = '#9aabbd';
            context.font = '9px "Segoe UI","Microsoft YaHei",sans-serif';
            context.textAlign = 'right';
            context.fillText(formatTime(time), side - 4, y - 2);
          }
        }
      }

      for (const note of chart.notes) {
        const start = note.start / 1000;
        const end = note.end / 1000;
        if (end < lowerTime || start > upperTime) continue;
        const x = side + note.lane * laneWidth + 2;
        const noteWidth = Math.max(2, laneWidth - 4);
        const headY = bottom - (start - current) * pixelsPerSecond;
        const laneColor = LANE_COLORS[note.lane % LANE_COLORS.length];
        if (note.hold) {
          const tailY = bottom - (end - current) * pixelsPerSecond;
          const y = Math.min(headY, tailY);
          const endY = Math.max(headY, tailY);
          context.fillStyle = laneColor;
          context.globalAlpha = 0.68;
          context.beginPath();
          context.roundRect(x + noteWidth * 0.18, y, noteWidth * 0.64, Math.max(5, endY - y), 4);
          context.fill();
          context.globalAlpha = 1;
          context.fillStyle = '#f3fcf8';
          context.beginPath();
          context.roundRect(x, headY - 4, noteWidth, 8, 4);
          context.fill();
          context.fillStyle = '#ffffff';
          context.fillRect(x, tailY - 2, noteWidth, 4);
        } else {
          context.fillStyle = laneColor;
          context.beginPath();
          context.roundRect(x, headY - 5, noteWidth, 10, 5);
          context.fill();
          context.fillStyle = '#ffffff';
          context.globalAlpha = 0.82;
          context.fillRect(x + noteWidth * 0.22, headY - 1, noteWidth * 0.56, 2);
          context.globalAlpha = 1;
        }
      }

      context.strokeStyle = '#fff4d6';
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(side, bottom);
      context.lineTo(side + keys * laneWidth, bottom);
      context.stroke();
      context.fillStyle = '#fff4d6';
      context.textAlign = 'left';
      context.font = '10px "Segoe UI","Microsoft YaHei",sans-serif';
      context.fillText(`${keys}K`, side + 4, 15);

      empty.hidden = chart.notes.length > 0;
      empty.textContent = chart.notes.length ? '' : '选择已生成的谱面后显示实时预览';
      updateProgress();
    }

    function tick() {
      raf = 0;
      if (destroyed || audio.paused || audio.ended) return;
      draw();
      raf = global.requestAnimationFrame(tick);
    }

    function startAnimation() {
      updateProgress();
      if (!raf && !audio.paused && !audio.ended) raf = global.requestAnimationFrame(tick);
      else draw();
    }

    playButton.addEventListener('click', () => {
      if (audio.paused || audio.ended) {
        if (audio.ended) audio.currentTime = 0;
        const pending = audio.play();
        if (pending && typeof pending.catch === 'function') pending.catch(() => updateProgress());
      } else {
        audio.pause();
      }
    }, { signal });
    progress.addEventListener('input', () => {
      const target = finite(progress.value);
      if (Number.isFinite(audio.duration)) audio.currentTime = Math.min(Math.max(0, target), audio.duration);
      else audio.currentTime = Math.max(0, target);
      draw();
    }, { signal });
    scrollInput.addEventListener('input', () => {
      scrollOutput.value = `${Number(scrollInput.value).toFixed(1)}x`;
      scrollOutput.textContent = scrollOutput.value;
      draw();
    }, { signal });
    rateInput.addEventListener('input', () => {
      const rate = Math.min(1.5, Math.max(0.5, finite(rateInput.value, 1)));
      audio.playbackRate = rate;
      rateOutput.value = `${rate.toFixed(1)}x`;
      rateOutput.textContent = rateOutput.value;
    }, { signal });
    audio.addEventListener('play', startAnimation, { signal });
    audio.addEventListener('playing', startAnimation, { signal });
    for (const name of ['pause', 'seeking', 'seeked', 'ratechange', 'timeupdate', 'ended', 'loadedmetadata']) {
      audio.addEventListener(name, () => {
        if (name === 'ratechange' && !rateInput.matches(':active')) {
          rateInput.value = String(Math.min(1.5, Math.max(0.5, finite(audio.playbackRate, 1))));
          rateOutput.value = `${Number(rateInput.value).toFixed(1)}x`;
          rateOutput.textContent = rateOutput.value;
        }
        if (name === 'pause' || name === 'ended') {
          if (raf) global.cancelAnimationFrame(raf);
          raf = 0;
        }
        draw();
      }, { signal });
    }
    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(resize);
      observer.observe(canvas);
    } else {
      global.addEventListener('resize', resize, { signal });
    }
    rateInput.value = String(Math.min(1.5, Math.max(0.5, finite(audio.playbackRate, 1))));
    rateOutput.value = `${Number(rateInput.value).toFixed(1)}x`;
    rateOutput.textContent = rateOutput.value;
    scrollOutput.value = `${Number(scrollInput.value).toFixed(1)}x`;
    scrollOutput.textContent = scrollOutput.value;
    resize();

    return {
      setChart(input) {
        const value = input && typeof input === 'object' ? input : {};
        const keys = Math.round(finite(value.keys, 10));
        if (keys < 6 || keys > 10) throw new RangeError('Preview key count must be between 6 and 10.');
        const bpm = finite(value.bpm, 120);
        if (bpm <= 0) throw new RangeError('Preview BPM must be positive.');
        chart = {
          notes: normalizeNotes(value.notes, keys),
          keys,
          bpm,
          offset: finite(value.offset, 0),
          division: Math.max(1, Math.min(32, Math.round(finite(value.division, 4)))),
          meter: Math.max(1, Math.min(12, Math.round(finite(value.meter, 4)))),
          stats: value.stats && typeof value.stats === 'object' ? value.stats : {},
        };
        renderStats();
        draw();
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        controller.abort();
        observer?.disconnect();
        if (raf) global.cancelAnimationFrame(raf);
        raf = 0;
        host.remove();
      },
    };
  }

  global.ManiaLabChartPreview = Object.freeze({ mount });
})(window);
