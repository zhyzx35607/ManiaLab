'use strict';
let editorNoteId=null;
function updateEditorLocks(){
  const v=variant(),strict=v?.mode==='remap',disabled=busy||!v;
  ['editStart','editEnd','editTap','editAdd','editWaveStart','editWaveEnd'].forEach(id=>$(id).disabled=disabled||strict);
  $('editLane').disabled=disabled;
  $('editSave').disabled=disabled||!editorNoteId;
  $('editDelete').disabled=disabled||strict||!editorNoteId;
}
function renderNoteEditor(note=null){
  const v=variant();editorNoteId=null;
  if(note&&view==='chart'&&v?.notes.some(n=>n.id===note.id)){
    editorNoteId=note.id;$('editStart').value=note.start;$('editEnd').value=note.end;$('editLane').value=note.lane+1;
    $('editorInfo').textContent=`已选择 ${note.end>note.start?'LN':'短键'}：${note.start}–${note.end} ms，第 ${note.lane+1} 轨。`+(v.mode==='remap'?'严格重定轨只允许换轨。':'起止将落到本候选的拍线，保存后显示实际结果。');
  }else $('editorInfo').textContent='在谱面预览中选择音符后修改；也可填写起止和轨道新增。尾端等于起点表示短键。';
  $('editLane').max=v?.settings.keys||keys;updateEditorLocks();
}
function editableVariant(){
  const v=variant();if(!v||busy)throw Error('请先生成候选谱面，等待当前任务完成');
  if(keys!==v.settings.keys)throw Error('当前键数与候选不同，请先重新生成或选回原键数');
  if(v.mode==='audio'&&(['bpm','offset','division'].some(k=>Number($(k).value)!==v[k])||clockShift()!==v.audio_shift_ms))throw Error('当前拍线已修改，请先重新生成整谱，再修正音符');
  return v;
}
async function submitNoteEdit(op){
  const v=editableVariant(),values={start:Number($('editStart').value),end:Number($('editEnd').value),lane:Number($('editLane').value)-1};
  if(op!=='add'&&!editorNoteId)throw Error('请在谱面预览中选中音符');
  const playhead=currentTime(),voice=$('voice').value,detailCursor=waveDetail?Number($('waveCursor').value):null;audio.pause();
  await runJob('edit',{op,note_id:editorNoteId,values,base_created:v.created});
  seek(playhead);
  if(voice!==$('voice').value&&!$('voice').disabled){$('voice').value=voice;$('voice').dispatchEvent(new Event('change'));}
  const edited=variant()?.edits?.at(-1)?.after;
  if(edited){setView('chart');inspectNote(edited);}
  $('noteEditor').open=true;
  if(detailCursor!==null)await loadWaveDetail(detailCursor);else window.drawWaveDetail?.();
}
$('editSave').onclick=guarded(()=>submitNoteEdit('update'));
$('editDelete').onclick=guarded(()=>submitNoteEdit('delete'));
$('editAdd').onclick=guarded(()=>submitNoteEdit('add'));
$('editTap').onclick=()=>{$('editEnd').value=$('editStart').value;};
for(const [button,field] of [['editWaveStart','editStart'],['editWaveEnd','editEnd']]){
  $(button).onclick=guarded(()=>{
    editableVariant();if(!waveDetail)throw Error('请先在波形对音中选中位置');
    // Convert the displayed waveform position to the candidate chart clock.
    $(field).value=Math.round(Number($('waveCursor').value)+waveDetail.shift-clockShift());
  });
}
renderNoteEditor();
