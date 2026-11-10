'use strict';
(() => {
  let available=false;
  const panel=document.createElement('div');
  const label=document.createElement('label');label.className='check';
  const check=document.createElement('input');check.type='checkbox';check.id='pianoSpecialist';
  label.append(check,document.createTextNode('钢琴声轨使用专用采音（测试）'));
  const hint=document.createElement('p');hint.className='muted compact';
  panel.append(label,hint);$('analysisEngine').closest('label').after(panel);
  const eligible=()=> $('analysisEngine').value==='deep6'||
    ($('analysisEngine').value==='provided'&&state?.provided_stems?.voices?.includes('piano'));
  window.lockPianoDetector=()=>{
    check.disabled=busy||!available||!eligible();
    hint.textContent=!available?'当前包未包含钢琴专用组件。':!eligible()?
      '选择六声部分离，或导入钢琴声轨后使用。':
      '仅用于钢琴声轨；使用固定检测参数。持续区间包含踏板延音，LN 表达仍由声部方案决定。';
  };
  window.updatePianoStatus=value=>{available=!!value?.available;window.lockPianoDetector();};
  window.renderPianoDetector=()=>{check.checked=state?.analysis?.piano_engine==='specialist';window.lockPianoDetector();};
  window.pianoSelection=()=>available&&eligible()&&check.checked?'specialist':'general';
  $('analysisEngine').addEventListener('change',window.lockPianoDetector);
  window.renderPianoDetector();
})();
