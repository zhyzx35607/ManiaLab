'use strict';
const lnProfiles={balanced:[.35,.65,0],release:[0,1,0],hybrid:[1,.5,0],inverse:[.35,.65,1]};
const lnControls=['ln_coordination','ln_release','ln_inverse'];

function updateLNProfile(){
  const values=lnControls.map(id=>Number($(id).value));
  $('lnPreset').value=Object.keys(lnProfiles).find(key=>lnProfiles[key].every((v,i)=>v===values[i]))||'custom';
  const classic=$('layoutEngine').value!=='beam';
  $('lnDesignHint').textContent=classic?'这些偏好用于连续键型搜索；选择方案会切回连续搜索。旧逐排引擎不使用它们。':'反键只使用已有 LN 尾端和下一起音；不会延长声音填空。放键间隔偏好仍生效。LN 比例与最大同时 LN 分别控制数量和占轨。';
}
$('lnPreset').onchange=()=>{
  const values=lnProfiles[$('lnPreset').value];if(!values)return;
  lnControls.forEach((id,i)=>$(id).value=values[i]);$('layoutEngine').value='beam';
  updateOutputs();updateLNProfile();msg('已应用 LN 键型偏好；生成后可查看实际配合统计，起止时间保持不变。');
};
lnControls.forEach(id=>$(id).addEventListener('input',updateLNProfile));
$('layoutEngine').addEventListener('change',updateLNProfile);

function renderLNDesign(){
  updateLNProfile();const v=variant(),r=v?.report?.ln_patterns;
  const metrics=$('lnDesignMetrics'),events=$('lnDesignEvents'),density=$('lnDensity');metrics.replaceChildren();events.replaceChildren();density.replaceChildren();
  if(!r){$('lnDesignSummary').textContent=v?'此候选来自旧版本，请重新定轨以查看完整的放手与协调分析。':'生成后显示实际 LN 占轨、放手、Hybrid 和反键；这些是键型特征，不是质量评分。';return;}
  const c=r.counts||{};
  $('lnDesignSummary').textContent=`LN 数量占比 ${(r.ln_ratio*100).toFixed(1)}%，占全曲轨道时间 ${(r.occupied_key_fraction*100).toFixed(1)}%。${r.notice}`;
  const entries=[['长按中短键',`${c.hybrid_taps||0} 个`],['其中同手配合',`${(r.same_hand_hybrid_fraction*100).toFixed(1)}%`],['反键接续',`${c.inverse_heads||0} 个`],['只有放手的时刻',`${c.release_only_rows||0} 处`],['峰值放手数 / 秒',r.peak_releases_per_second],['左右手最大占轨',r.peak_held_by_hand.join(' / ')]];
  for(const [label,value] of entries){const item=document.createElement('div'),strong=document.createElement('strong'),small=document.createElement('small');strong.textContent=value;small.textContent=label;item.append(strong,small);metrics.append(item);}
  for(const part of (r.sections||[]).filter(s=>Number.isFinite(s.occupied_key_fraction)&&s.end_ms>s.start_ms)){
    const button=document.createElement('button'),bar=document.createElement('span'),label=document.createElement('small');
    button.className='ln-density-part';bar.style.height=`${Math.max(2,part.occupied_key_fraction*100)}%`;label.textContent=fmt(part.start_ms/1000);
    button.title=`${fmt(part.start_ms/1000)}–${fmt(part.end_ms/1000)} · 占轨时间 ${(part.occupied_key_fraction*100).toFixed(1)}% · 最大同时 ${part.peak_held} LN · ${part.counts.releases||0} 次放手`;
    button.setAttribute('aria-label',button.title);button.append(bar,label);
    button.onclick=()=>{audio.pause();setView('chart');seek(part.start_ms/1000);$('auditionStart').value=(part.start_ms/1000).toFixed(1);$('lnDensityInfo').textContent=button.title;};density.append(button);
  }
  const selectedKind=$('lnEventFilter').value,locations=r.locations.filter(x=>x.kind===selectedKind);
  for(const item of locations.slice(0,20)){
    const button=document.createElement('button');button.className='quiet';button.textContent=`${fmt(item.start_ms/1000)} · ${item.text}`;
    button.onclick=guarded(async()=>{audio.pause();setView('chart');seek(Math.max(0,item.start_ms/1000-.6));$('auditionStart').value=Math.max(0,item.start_ms/1000-1).toFixed(1);const n=v.notes.find(n=>item.columns.includes(n.lane)&&(n.end===item.start_ms||n.start===item.start_ms));if(n)inspectNote(n);if(state.analysis)await loadWaveDetail(item.start_ms);});events.append(button);
  }
  if(!locations.length){const p=document.createElement('p');p.className='muted';p.textContent=c[selectedKind]?'此旧候选未保留该类定位样本，重新定轨可按全曲分段定位。':'当前谱面没有这一类事件。';events.append(p);}
}
$('lnEventFilter').onchange=renderLNDesign;
renderLNDesign();
