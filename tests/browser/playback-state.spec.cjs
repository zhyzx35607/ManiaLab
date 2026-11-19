const {test,expect}=require('@playwright/test');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs');
const http=require('node:http');
const path=require('node:path');

async function openSourceProject(page){
  await page.goto('/');
  await expect(page.locator('#recent option')).toContainText(['UI evidence source identity']);
  await page.locator('#recent').selectOption({label:'UI evidence source identity'});
  await expect(page.locator('#projectTitle')).toHaveText('UI evidence source identity');
  if(await page.locator('#variants').inputValue()!=='0')await page.locator('#variants').selectOption('0');
  await expect.poll(()=>page.evaluate(()=>state.active)).toBe(0);
  await page.waitForFunction(()=>audio.readyState>=audio.HAVE_METADATA);
  await page.locator('#audio').evaluate(player=>{player.muted=true;player.currentTime=3;});
  await expect.poll(()=>page.locator('#audio').evaluate(player=>player.currentTime)).toBeCloseTo(3,1);
}

async function mediaGate(page){
  let release,firstReady,nextReady,count=0;
  const gate=new Promise(resolve=>{release=resolve;});
  const first=new Promise(resolve=>{firstReady=resolve;});
  const next=new Promise(resolve=>{nextReady=resolve;});
  await page.route(url=>url.pathname.startsWith('/api/projects/')&&
    (url.pathname.includes('/stem/')||url.pathname.endsWith('/audio')),async route=>{
      if(++count===1)firstReady();else nextReady();
      await gate;
      try{await route.continue();}catch{}
    });
  return {release,first,next};
}

async function releaseMedia(page,gate){
  const loaded=page.evaluate(()=>new Promise(resolve=>{
    if(audio.readyState>=audio.HAVE_METADATA){resolve(true);return;}
    audio.addEventListener('loadedmetadata',()=>resolve(true),{once:true});
  }));
  gate.release();await loaded;
  await page.waitForFunction(()=>!audio.seeking);
}

for(const kind of ['voice','view','candidate','project']){
  test('paused playback keeps the intended position during a rapid '+kind+' switch',async({page})=>{
    await openSourceProject(page);
    let latest=0;
    if(kind==='candidate'){
      await page.locator('#timingConfirmed').check();
      await page.locator('#generate').click();
      await expect(page.locator('#message')).toContainText('结构检查通过');
      latest=await page.evaluate(()=>state.active);
      expect(latest).toBeGreaterThan(0);
      await page.locator('#variants').selectOption('0');
      await expect.poll(()=>page.evaluate(()=>state.active)).toBe(0);
      await page.locator('#audio').evaluate(player=>{player.currentTime=3;});
      await expect.poll(()=>page.locator('#audio').evaluate(player=>player.currentTime)).toBeCloseTo(3,1);
    }
    const gate=await mediaGate(page);
    try{
      await page.locator('#voice').selectOption('other');await gate.first;
      expect(await page.locator('#audio').evaluate(player=>player.readyState)).toBe(0);
      if(kind==='voice')await page.locator('#voice').selectOption('piano');
      else if(kind==='view')await page.locator('#evidenceView').click();
      else if(kind==='candidate'){
        await page.locator('#variants').selectOption(String(latest));
        await expect.poll(()=>page.evaluate(()=>state.active)).toBe(latest);
      }else{
        await page.locator('#recent').selectOption({label:'UI timing wave'});
        await expect(page.locator('#projectTitle')).toHaveText('UI timing wave');
      }
      await gate.next;await releaseMedia(page,gate);
      await expect.poll(()=>page.locator('#audio').evaluate(player=>player.currentTime)).toBeCloseTo(kind==='project'?0:3,1);
      expect(await page.locator('#audio').evaluate(player=>player.paused)).toBe(true);
      expect(await page.evaluate(()=>audio.getAttribute('src'))).toBe(await page.evaluate(()=>playbackSourceUrl()));
    }finally{gate.release();}
  });
}

test('an explicit pause during media loading cancels the pending playback intent',async({page})=>{
  await openSourceProject(page);
  await page.locator('#audio').evaluate(player=>player.play());
  await expect.poll(()=>page.locator('#audio').evaluate(player=>player.paused)).toBe(false);
  const gate=await mediaGate(page);
  try{
    await page.locator('#voice').selectOption('other');await gate.first;
    await expect.poll(()=>page.locator('#audio').evaluate(player=>player.paused)).toBe(false);
    await page.locator('#audio').evaluate(player=>player.pause());
    expect(await page.locator('#audio').evaluate(player=>player.paused)).toBe(true);
    await releaseMedia(page,gate);
    expect(await page.locator('#audio').evaluate(player=>player.paused)).toBe(true);
    await expect.poll(()=>page.locator('#audio').evaluate(player=>player.currentTime)).toBeCloseTo(3,1);
    await expect(page.locator('#message')).not.toContainText('音频播放失败');
  }finally{gate.release();}
});

test('the latest source reports a native playback rejection instead of implying success',async({page})=>{
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  await openSourceProject(page);
  await page.locator('#audio').evaluate(player=>player.play());
  // Only the media bytes fail; the native decoder and playback promise remain real.
  await page.route('**/stem/other?*',route=>route.fulfill({status:200,
    contentType:'audio/wav',body:'synthetic unreadable audio'}));
  await page.locator('#voice').selectOption('other');
  await expect(page.locator('#message')).toContainText('音频播放失败');
  expect(await page.locator('#message').evaluate(element=>element.classList.contains('error'))).toBe(true);
  expect(await page.locator('#audio').evaluate(player=>player.paused)).toBe(true);
  expect(await page.locator('#audio').evaluate(player=>player.error?.code)).toBe(4);
  expect(errors).toEqual([]);
});

for(const rapid of [false,true]){
  test('continuous playback resumes the latest '+(rapid?'rapid':'single')+' source at its saved position',async({page})=>{
    await openSourceProject(page);
    await page.locator('#audio').evaluate(player=>player.play());
    await expect.poll(()=>page.locator('#audio').evaluate(player=>player.paused)).toBe(false);
    const gate=await mediaGate(page);
    try{
      await page.locator('#voice').selectOption('other');await gate.first;
      if(rapid){await page.locator('#voice').selectOption('piano');await gate.next;}
      await releaseMedia(page,gate);
      await page.waitForFunction(()=>!audio.paused&&audio.readyState>=audio.HAVE_FUTURE_DATA&&audio.currentTime>3.04);
      expect(await page.locator('#audio').evaluate(player=>player.currentTime)).toBeLessThan(5);
      expect(await page.evaluate(()=>audio.getAttribute('src'))).toBe(await page.evaluate(()=>playbackSourceUrl()));
      await page.locator('#audio').evaluate(player=>player.pause());
    }finally{gate.release();}
  });
}

test('pending playback without metadata cannot schedule cue oscillators',async({page})=>{
  await openSourceProject(page);
  await page.locator('#offset').fill('0');
  await page.locator('#metronome').check();
  await page.waitForFunction(()=>cueClock.context?.state==='running');
  await page.locator('#audio').evaluate(player=>player.play());
  await page.evaluate(()=>{
    window.loadingCueStarts=[];window.loadingScheduleCalls=0;
    const create=cueClock.context.createOscillator.bind(cueClock.context);
    cueClock.context.createOscillator=(...args)=>{
      const node=create(...args),start=node.start.bind(node);
      node.start=(...values)=>{window.loadingCueStarts.push(audio.readyState);return start(...values);};
      return node;
    };
    const schedule=cueClock.schedule.bind(cueClock);
    cueClock.schedule=()=>{if(audio.readyState===0)window.loadingScheduleCalls++;return schedule();};
  });
  const gate=await mediaGate(page);
  try{
    await page.locator('#voice').selectOption('other');await gate.first;
    await page.evaluate(()=>{window.loadingCueStarts=[];});
    await page.waitForFunction(()=>window.loadingScheduleCalls>=2);
    expect(await page.evaluate(()=>window.loadingCueStarts)).toEqual([]);
    expect(await page.evaluate(()=>cueClock.nodes.size)).toBe(0);
    await releaseMedia(page,gate);
    await page.waitForFunction(()=>window.loadingCueStarts.some(value=>value>=audio.HAVE_FUTURE_DATA));
    await page.locator('#audio').evaluate(player=>player.pause());
    expect(await page.evaluate(()=>cueClock.nodes.size)).toBe(0);
  }finally{gate.release();}
});

test('real buffering cancels scheduled cues and pause or seek releases their nodes',async({page},testInfo)=>{
  const root=path.resolve(__dirname,'../..');
  const python=process.env.MANIALAB_TEST_PYTHON||path.join(root,'.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
  const mediaPath=testInfo.outputPath('buffering-synthetic.mp3');
  fs.mkdirSync(path.dirname(mediaPath),{recursive:true});
  execFileSync(python,['-c',[
    'import imageio_ffmpeg, subprocess, sys;',
    'subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",',
    '"sine=frequency=440:duration=20", "-c:a", "libmp3lame", "-b:a", "128k", sys.argv[1]], check=True)',
  ].join(' '),mediaPath],{cwd:root,windowsHide:true});
  const bytes=fs.readFileSync(mediaPath),firstChunk=42000;
  const clockSource=fs.readFileSync(path.join(root,'static/audio-clock.js'),'utf8');
  const partials=new Set();
  let releaseStream;
  const server=http.createServer((request,response)=>{
    if(request.url==='/stream.mp3'){
      const start=Number((request.headers.range||'').split('=')[1]?.split('-')[0]||0);
      const headers={'Content-Type':'audio/mpeg','Content-Length':bytes.length-start,
        'Accept-Ranges':'bytes','Cache-Control':'no-store'};
      if(request.headers.range)headers['Content-Range']='bytes '+start+'-'+(bytes.length-1)+'/'+bytes.length;
      response.writeHead(request.headers.range?206:200,headers);
      if(start>0){response.end(bytes.subarray(start));return;}
      response.write(bytes.subarray(0,firstChunk));
      partials.add(response);response.on('close',()=>partials.delete(response));
      releaseStream=()=>response.end(bytes.subarray(firstChunk));
    }else{
      response.writeHead(200,{'Content-Type':'text/html'});
      response.end(['<audio id="streamMedia" src="/stream.mp3" preload="auto"></audio>',
        '<button id="streamPlay">Play</button><script>',clockSource,'</script>'].join(String.fromCharCode(10)));
    }
  });
  const errors=[];page.on('pageerror',error=>errors.push(error.message));
  try{
    await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
    await page.goto('http://127.0.0.1:'+server.address().port,{waitUntil:'domcontentloaded'});
    await page.evaluate(()=>{
      const media=document.getElementById('streamMedia');
      window.streamEvents=[];window.streamCues=[];window.streamGains=new Set();
      window.streamClock=new ManiaAudioClock(media,(from,to)=>{
        const events=[];
        for(let tick=Math.ceil(from/.05);tick*.05<to;tick++)events.push({time:tick*.05,kind:'beat'});
        return events;
      });
      for(const type of ['waiting','playing'])media.addEventListener(type,()=>window.streamEvents.push({
        type,mediaTime:media.currentTime,readyState:media.readyState,
        contextTime:window.streamClock.context?.currentTime,nodes:window.streamClock.nodes.size,
        scheduled:window.streamClock.scheduled.size,
      }));
      document.getElementById('streamPlay').onclick=async()=>{
        const clock=window.streamClock;
        await clock.enable();
        const create=clock.context.createOscillator.bind(clock.context);
        clock.context.createOscillator=(...args)=>{
          const node=create(...args),row={stops:[]};window.streamCues.push(row);
          const start=node.start.bind(node),stop=node.stop.bind(node),disconnect=node.disconnect.bind(node);
          node.start=(at)=>{row.at=at;row.startMedia=media.currentTime;return start(at);};
          node.stop=(...values)=>{row.stops.push({when:values[0]??null,calledAt:clock.context.currentTime});return stop(...values);};
          node.disconnect=(...values)=>{row.disconnected=true;return disconnect(...values);};
          node.addEventListener('ended',()=>{row.endedContext=clock.context.currentTime;});
          return node;
        };
        const gain=clock.context.createGain.bind(clock.context);
        clock.context.createGain=(...args)=>{
          const node=gain(...args),disconnect=node.disconnect.bind(node);window.streamGains.add(node);
          node.disconnect=(...values)=>{window.streamGains.delete(node);return disconnect(...values);};
          return node;
        };
        await media.play();
      };
    });
    await page.locator('#streamPlay').click();
    await page.waitForFunction(()=>window.streamEvents.some(row=>row.type==='waiting'&&row.mediaTime>.2));
    const waiting=await page.evaluate(()=>window.streamEvents.find(row=>row.type==='waiting'&&row.mediaTime>.2));
    expect(waiting.readyState).toBeLessThan(3);
    expect(waiting.nodes).toBe(0);expect(waiting.scheduled).toBe(0);
    expect(await page.evaluate(at=>window.streamCues.some(row=>row.at>at&&
      row.stops.some(stop=>stop.when===null&&stop.calledAt<row.at)),waiting.contextTime)).toBe(true);
    await page.waitForFunction(()=>window.streamCues.every(row=>row.disconnected));
    expect(await page.evaluate(()=>window.streamGains.size)).toBe(0);
    const oldCues=await page.evaluate(()=>window.streamCues.length);
    releaseStream();
    await page.waitForFunction(({count,time})=>{
      const media=document.getElementById('streamMedia');
      return media.readyState>=media.HAVE_FUTURE_DATA&&media.currentTime>time+.1&&window.streamCues.length>count;
    },{count:oldCues,time:waiting.mediaTime});
    const seeking=await page.evaluate(()=>{
      const media=document.getElementById('streamMedia'),clock=window.streamClock;
      const result=new Promise(resolve=>media.addEventListener('seeking',()=>resolve({
        nodes:clock.nodes.size,scheduled:clock.scheduled.size,
      }),{once:true}));
      media.currentTime=4;return result;
    });
    expect(seeking).toEqual({nodes:0,scheduled:0});
    await page.waitForFunction(()=>!document.getElementById('streamMedia').seeking&&!document.getElementById('streamMedia').paused);
    const paused=await page.evaluate(()=>{
      const media=document.getElementById('streamMedia'),clock=window.streamClock;
      const result=new Promise(resolve=>media.addEventListener('pause',()=>resolve({
        nodes:clock.nodes.size,scheduled:clock.scheduled.size,
      }),{once:true}));
      media.pause();return result;
    });
    expect(paused).toEqual({nodes:0,scheduled:0});
    await page.evaluate(()=>{window.streamClock.reset();window.streamClock.reset();});
    await page.waitForFunction(()=>window.streamCues.every(row=>row.disconnected));
    expect(await page.evaluate(()=>window.streamGains.size)).toBe(0);
    await testInfo.attach('native-buffering-events',{body:JSON.stringify(await page.evaluate(()=>({
      events:window.streamEvents,cues:window.streamCues,
    })),null,2),contentType:'application/json'});
    expect(errors).toEqual([]);
  }finally{
    try{await page.goto('about:blank');}
    finally{
      for(const response of partials)response.destroy();
      await new Promise(resolve=>server.close(resolve));
    }
  }
});
