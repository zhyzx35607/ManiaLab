'use strict';

// Music and cues share one AudioContext output. Scheduling is ahead of the
// playhead; animation frames are used for drawing only.
class ManiaAudioClock {
  constructor(media, events) {
    this.media=media;this.events=events;this.context=null;this.nodes=new Set();this.scheduled=new Set();
    this.timer=null;
    for(const name of ['pause','seeking','ratechange','ended'])media.addEventListener(name,()=>this.reset());
    media.addEventListener('play',()=>{if(this.context)this.enable();});
    media.addEventListener('seeked',()=>{this.reset();this.schedule();});
  }
  async enable() {
    if(!this.context){
      this.context=new(window.AudioContext||window.webkitAudioContext)({latencyHint:'interactive'});
      this.source=this.context.createMediaElementSource(this.media);this.source.connect(this.context.destination);
    }
    await this.context.resume();
    if(!this.timer)this.timer=setInterval(()=>this.schedule(),25);
    this.reset();this.schedule();
  }
  reset() {
    for(const node of this.nodes){try{node.stop();}catch{}}
    this.nodes.clear();this.scheduled.clear();
  }
  schedule() {
    const ctx=this.context,media=this.media;
    if(!ctx||ctx.state!=='running'||media.paused||media.seeking||media.ended)return;
    const now=media.currentTime,rate=media.playbackRate;
    for(const e of this.events(now,now+.14*rate)){
      if(e.time<now||e.time>now+.14*rate)continue;
      const key=`${e.time.toFixed(6)}:${e.kind}`;
      if(this.scheduled.has(key))continue;
      const at=ctx.currentTime+(e.time-now)/rate;
      const oscillator=ctx.createOscillator(),gain=ctx.createGain();
      oscillator.frequency.value={accent:1600,beat:1150,head:900,tail:550}[e.kind]||900;
      gain.gain.setValueAtTime(.045,at);gain.gain.exponentialRampToValueAtTime(.001,at+.03);
      oscillator.connect(gain).connect(ctx.destination);oscillator.start(at);oscillator.stop(at+.035);
      this.nodes.add(oscillator);this.scheduled.add(key);
      oscillator.onended=()=>{this.nodes.delete(oscillator);oscillator.disconnect();gain.disconnect();};
    }
    // Keep keys for completed events briefly, without an unbounded history.
    if(this.scheduled.size>2000)this.scheduled=new Set([...this.scheduled].filter(k=>Number(k.split(':')[0])>=now-.5));
  }
}
