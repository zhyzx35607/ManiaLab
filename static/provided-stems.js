'use strict';
(() => {
  const names={drums:'鼓 / 打击',bass:'贝斯',vocals:'人声',other:'其余伴奏',guitar:'吉他',piano:'钢琴'};
  const panel=document.createElement('details');panel.className='voice-preferences';panel.id='providedStemsPanel';
  const summary=document.createElement('summary');summary.textContent='使用已有声轨 · Nova 等工具的分离结果';panel.append(summary);
  const hint=document.createElement('p');hint.className='muted compact';
  hint.textContent='先导入原曲，再按乐器选择声轨。每次保存只使用本次选中的声部；可以只选钢琴或鼓。请保留开头静音，声轨应与原曲同起点、同长度。';panel.append(hint);
  for(const [voice,name] of Object.entries(names)){
    const label=document.createElement('label');label.className='field';label.textContent=name;
    const input=document.createElement('input');input.type='file';input.accept='audio/*';input.id='provided-'+voice;input.dataset.providedVoice=voice;
    label.append(input);panel.append(label);
  }
  const save=document.createElement('button');save.className='secondary full';save.id='providedStemsSave';save.textContent='保存这些声轨';panel.append(save);
  const status=document.createElement('p');status.className='muted compact';status.id='providedStemsStatus';panel.append(status);
  $('analysisEngine').closest('label').after(panel);
  const option=new Option('使用已导入声轨：逐乐器采音','provided');option.disabled=true;$('analysisEngine').add(option);
  window.renderProvidedStems=()=>{
    const supplied=state?.provided_stems;
    option.disabled=!supplied;
    if(!supplied&&$('analysisEngine').value==='provided')$('analysisEngine').value='basic';
    status.textContent=supplied?`已保存：${supplied.voices.map(v=>names[v]).join('、')} · ${supplied.duration_seconds.toFixed(2)} 秒。选择「使用已导入声轨」后分析，再逐声部核对起音。`:'尚未导入独立声轨。';
    window.lockProvidedStems();
  };
  window.lockProvidedStems=()=>{
    save.disabled=busy||!state?.audio;
    panel.querySelectorAll('input').forEach(input=>input.disabled=busy||!state?.audio);
  };
  save.onclick=guarded(async()=>{
    if(!state?.audio)throw Error('请先导入原曲');
    const data=new FormData();let count=0;
    for(const input of panel.querySelectorAll('input'))if(input.files[0]){data.append(input.dataset.providedVoice,input.files[0]);count++;}
    if(!count)throw Error('至少选择一个声部文件');
    const outcome=await runJob('provided-stems',data,true);
    if(outcome==='done'){$('analysisEngine').value='provided';$('analysisEngine').dispatchEvent(new Event('change'));panel.querySelectorAll('input').forEach(input=>input.value='');}
  });
  window.renderProvidedStems();
})();
