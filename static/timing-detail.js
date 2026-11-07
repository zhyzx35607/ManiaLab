'use strict';
let waveDetail=null,waveDetailRequest=0;
const waveShift=()=>$('waveClock').checked?clockShift():0;

function resetWaveDetail(){waveDetailRequest++;waveDetail=null;drawWaveDetail();}
async function loadWaveDetail(timeMs=Math.max(0,currentTime()*1000-waveShift())){
  if(!state?.analysis)throw Error('请先分析音频，再查看波形细节');
  const requestNumber=++waveDetailRequest,pid=state.id,voice=$('voice').value;
  const span=Number($('waveSpan').value),shift=waveShift(),start=Math.max(0,timeMs+shift-span*.25);
  $('waveDetailPanel').open=true;$('waveDetailInfo').textContent='正在读取原始采样波形…';
  const data=await api(`/api/projects/${pid}/waveform?start_ms=${start}&duration_ms=${span}&voice=${voice}`);
  if(state?.id!==pid||requestNumber!==waveDetailRequest)return;
  waveDetail={...data,voice,shift,start_ms:data.start_ms-shift,end_ms:data.end_ms-shift};$('waveCursor').value=Math.round(timeMs);drawWaveDetail();
}
function drawWaveDetail(){
  const [c,w,h]=canvas('waveDetail');c.clearRect(0,0,w,h);
  if(!waveDetail){c.fillStyle='#8d9db5';c.font='12px sans-serif';c.fillText('查看当前播放位置，或选中一个音符后放大。',12,45);return;}
  const d=waveDetail,left=12,right=w-12,width=right-left,span=d.end_ms-d.start_ms;
  const x=t=>left+(t-d.start_ms)/span*width;
  const line=(t,color,top=27,bottom=h-27,dashed=false)=>{if(t<d.start_ms||t>d.end_ms)return;c.strokeStyle=color;c.setLineDash(dashed?[3,3]:[]);c.beginPath();c.moveTo(x(t),top);c.lineTo(x(t),bottom);c.stroke();c.setLineDash([]);};
  c.fillStyle='#102030';c.fillRect(left,25,width,h-50);
  if($('mode').value==='audio'){
    const bpm=Number($('bpm').value),offset=Number($('offset').value),division=Number($('division').value),tick=60000/bpm/division;
    if(tick>0&&Number.isFinite(tick))for(let i=Math.ceil((d.start_ms-offset)/tick),n=0;offset+i*tick<=d.end_ms&&n<500;i++,n++){
      const t=offset+i*tick,beat=i%division===0;line(t,beat?'#6b8c82':'#334a4d');
      if(beat&&tick*division/span*width>42){c.fillStyle='#a6beb6';c.font='10px monospace';c.fillText(`${Math.round(t)}`,x(t)+3,16);}
    }
  }else for(const b of beatsBetween(d.start_ms/1000,d.end_ms/1000))line(b.time*1000,'#6b8c82');
  const peak=Math.max(.001,...d.high.map(Math.abs),...d.low.map(Math.abs)),mid=h*.48,scale=(h-78)*.42/peak;
  c.strokeStyle='#8bb8ff';c.beginPath();
  for(let i=0;i<d.low.length;i++){
    const at=x(d.start_ms+i*d.step_samples*1000/d.sample_rate);
    c.moveTo(at,mid-d.low[i]*scale);c.lineTo(at,mid-d.high[i]*scale);
  }c.stroke();
  const relevant=n=>d.voice==='mix'||n.voice===d.voice;
  for(const n of (state.analysis?.candidates||[]).filter(relevant))line(n.start-d.shift,'#ffb978',h-54,h-34);
  for(const n of (variant()?.notes||state.input_notes||[]).filter(relevant)){
    line(n.start,'#eef4ff',h-32,h-14);if(n.end>n.start)line(n.end,'#78e4c4',27,h-14,true);
  }
  const cursor=Number($('waveCursor').value);line(cursor,'#ff839d',0,h-10);
  c.fillStyle='#8d9db5';c.font='10px monospace';c.fillText(`${d.start_ms.toFixed(1)} ms`,left,h-2);c.fillText(`${d.end_ms.toFixed(1)} ms`,Math.max(left,right-85),h-2);
  const tick=60000/Number($('bpm').value)/Number($('division').value),offset=Number($('offset').value);
  const delta=cursor-(offset+Math.round((cursor-offset)/tick)*tick);
  $('waveAlign').disabled=busy||$('mode').value!=='audio';
  $('waveDetailInfo').textContent=`${d.voice} · 原始采样 ${d.sample_rate} Hz · 波形显示提前 ${d.shift} ms · 选中 ${cursor.toFixed(0)} ms`+($('mode').value==='audio'?`，距最近细分线 ${delta.toFixed(2)} ms。`:'。')+'橙色：原始起音；白色：谱面起点；绿色虚线：LN 尾端。波形位置需要结合试听判断。';
}
$('waveHere').onclick=guarded(()=>loadWaveDetail());
$('waveNote').onclick=guarded(()=>{const note=displayed.find(n=>n.id===selected);if(!note)throw Error('请先在谱面或采音预览中点击一个音符');return loadWaveDetail(note.start-(view==='evidence'?waveShift():0));});
$('waveTail').onclick=guarded(()=>{const note=displayed.find(n=>n.id===selected);if(!note||note.end<=note.start)throw Error('请先在谱面或采音预览中点击一条 LN / 持续音');return loadWaveDetail(note.end-(view==='evidence'?waveShift():0));});
$('waveSpan').onchange=guarded(()=>loadWaveDetail(waveDetail?Number($('waveCursor').value):currentTime()*1000));
$('wavePrev').onclick=guarded(()=>loadWaveDetail(Math.max(0,(waveDetail?.start_ms||0)-Number($('waveSpan').value)*.5)));
$('waveNext').onclick=guarded(()=>loadWaveDetail(Math.min(duration()*1000-100,(waveDetail?.end_ms||0))));
$('waveCursor').oninput=drawWaveDetail;
$('waveDetail').onclick=e=>{if(!waveDetail)return;const rect=$('waveDetail').getBoundingClientRect(),fraction=Math.max(0,Math.min(1,(e.clientX-rect.left-12)/(rect.width-24)));$('waveCursor').value=Math.round(waveDetail.start_ms+fraction*(waveDetail.end_ms-waveDetail.start_ms));drawWaveDetail();};
$('waveAlign').onclick=guarded(()=>{
  if(!waveDetail||$('mode').value!=='audio')throw Error('请先打开音频生成模式的波形片段');
  const cursor=Number($('waveCursor').value),offset=Number($('offset').value),tick=60000/Number($('bpm').value)/Number($('division').value);
  if(!Number.isFinite(cursor)||!Number.isFinite(tick)||tick<=0)throw Error('请填写有效 BPM 和选中时间');
  const change=cursor-(offset+Math.round((cursor-offset)/tick)*tick);
  $('offset').value=Math.round(offset+change);timingChanged();
  msg(`Offset 调整为 ${$('offset').value} ms。请在歌曲多个位置试听确认，再重新生成整谱。`);
});
$('voice').addEventListener('change',()=>{resetWaveDetail();if($('waveDetailPanel').open)guarded(()=>loadWaveDetail())();});
$('mode').addEventListener('change',drawWaveDetail);
$('waveDetailPanel').addEventListener('toggle',drawWaveDetail);
window.addEventListener('resize',drawWaveDetail);drawWaveDetail();

$('waveClock').addEventListener('change',guarded(()=>{resetWaveDetail();return loadWaveDetail();}));
$('audioShift').addEventListener('change',()=>{resetWaveDetail();if($('waveDetailPanel').open)guarded(()=>loadWaveDetail())();});
