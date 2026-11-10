'use strict';
(() => {
  let available=false;
  const panel=document.createElement('div');panel.id='guitarDetector';
  const makeCheck=(id,text)=>{
    const label=document.createElement('label');label.className='check';
    const input=document.createElement('input');input.type='checkbox';input.id=id;
    label.append(input,document.createTextNode(text));panel.append(label);return input;
  };
  const selected=makeCheck('guitarSpecialist','吉他声轨使用专用采音（测试）');
  const releases=makeCheck('guitarPredictedReleases','允许吉他模型音尾参与长条');
  const hint=document.createElement('p');hint.className='muted compact';panel.append(hint);
  $('analysisEngine').closest('label').after(panel);
  const eligible=()=> $('analysisEngine').value==='deep6'||
    ($('analysisEngine').value==='provided'&&state?.provided_stems?.voices?.includes('guitar'));
  window.lockGuitarDetector=()=>{
    selected.disabled=busy||!available||!eligible();
    releases.disabled=selected.disabled||!selected.checked;
    hint.textContent=!available?'当前包未包含吉他专用采音模型。':!eligible()?
      '选择六声部分离，或导入吉他声轨后使用。':
      selected.checked&&releases.checked?
      '允许有来源支持的吉他持续区间参与长条；重弹截断等未决音尾仍保持短键。请试听检查起音与放手位置。':
      '默认只使用吉他专用模型的起音。需要用模型音尾做长条时，另勾选上方吉他音尾选项。';
  };
  window.guitarSelection=()=>{
    const enabled=available&&eligible()&&selected.checked;
    return {guitar_engine:enabled?'fl':'general',allow_guitar_predicted_releases:!!(enabled&&releases.checked)};
  };
  window.updateGuitarStatus=value=>{available=!!value?.available;window.lockGuitarDetector();};
  window.renderGuitarDetector=()=>{
    const saved=state?.analysis_options??state?.analysis;
    selected.checked=saved?.guitar_engine==='fl';
    releases.checked=selected.checked&&saved?.allow_guitar_predicted_releases===true;
    window.lockGuitarDetector();
  };
  selected.addEventListener('change',()=>{if(!selected.checked)releases.checked=false;window.lockGuitarDetector();});
  releases.addEventListener('change',window.lockGuitarDetector);
  $('analysisEngine').addEventListener('change',window.lockGuitarDetector);
  window.renderGuitarDetector();
})();
