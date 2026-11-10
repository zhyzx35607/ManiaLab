'use strict';
(() => {
  let available=false;
  const panel=document.createElement('div');panel.id='instrumentDetector';
  const makeCheck=(id,text)=>{
    const label=document.createElement('label');label.className='check';
    const input=document.createElement('input');input.type='checkbox';input.id=id;
    label.append(input,document.createTextNode(text));panel.append(label);return input;
  };
  const selected=makeCheck('instrumentSpecialist','贝斯与其他伴奏专用采音（测试）');
  const releases=makeCheck('instrumentPredictedReleases','允许模型预测音尾参与长条');
  const hint=document.createElement('p');hint.className='muted compact';panel.append(hint);
  $('analysisEngine').closest('label').after(panel);
  const eligible=()=>['deep','deep6'].includes($('analysisEngine').value)||
    ($('analysisEngine').value==='provided'&&state?.provided_stems?.voices?.some(v=>['bass','other'].includes(v)));
  window.lockInstrumentDetector=()=>{
    selected.disabled=busy||!available||!eligible();
    releases.disabled=selected.disabled||!selected.checked;
    hint.textContent=!available?'当前包未包含贝斯与其他伴奏专用组件。':!eligible()?
      '选择四／六声部分离，或导入贝斯、其他伴奏声轨后使用。':
      releases.checked&&selected.checked?
      '将尝试使用有来源支持的模型音尾；未决尾仍保持短键。音色标签和放手位置都需要试听确认。':
      '默认只把专用模型的起音交给编排。需要用这些模型音尾做长条时，另勾选上方音尾选项；未决尾始终不参与长条。';
  };
  window.instrumentSelection=()=>{
    const enabled=available&&eligible()&&selected.checked;
    return {instrument_engine:enabled?'tsumugi':'general',allow_predicted_releases:!!(enabled&&releases.checked)};
  };
  window.updateInstrumentStatus=value=>{available=!!value?.available;window.lockInstrumentDetector();};
  window.renderInstrumentDetector=()=>{
    selected.checked=state?.analysis?.instrument_engine==='tsumugi';
    releases.checked=selected.checked&&state?.analysis?.allow_predicted_releases===true;
    window.lockInstrumentDetector();
  };
  selected.addEventListener('change',()=>{if(!selected.checked)releases.checked=false;window.lockInstrumentDetector();});
  releases.addEventListener('change',window.lockInstrumentDetector);
  $('analysisEngine').addEventListener('change',window.lockInstrumentDetector);
  window.renderInstrumentDetector();
})();
