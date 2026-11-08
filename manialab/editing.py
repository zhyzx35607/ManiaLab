"""Explicit, validated corrections saved as new candidates, never over evidence."""
from dataclasses import replace
import secrets

from .domain import Note, note_from, validate
from .timing import align_candidates


def edit_notes(variant, payload, duration_ms=None):
    if not isinstance(payload, dict) or payload.get('op') not in ('update','delete','add'):
        raise ValueError('编辑操作无效')
    op = payload['op']
    notes = [note_from(n) for n in variant['notes']]
    strict = variant['mode'] == 'remap'
    if strict and op != 'update':
        raise ValueError('严格重定轨只允许修改轨道，不能增删物件或改变起止')
    if not strict and variant.get('timing_revision',0) < 3:
        raise ValueError('旧版候选请先重新生成，再进行编辑')
    old = None
    if op != 'add':
        matches = [n for n in notes if n.id == payload.get('note_id')]
        if len(matches) != 1:
            raise ValueError('选中的音符已改变，请重新选择')
        old = matches[0]
    changes = payload.get('values',{})
    if not isinstance(changes,dict) or set(changes)-{'start','end','lane'}:
        raise ValueError('只可编辑起点、尾端和轨道')
    if op == 'delete':
        result = [n for n in notes if n is not old]
        new = None
    else:
        for key,value in changes.items():
            if type(value) is not int:
                raise ValueError('音符时间和轨道必须是整数')
        if strict and any(changes.get(key,getattr(old,key)) != getattr(old,key) for key in ('start','end')):
            raise ValueError('严格重定轨只允许修改轨道，不能改变起止')
        if op == 'add' and set(changes) != {'start','end','lane'}:
            raise ValueError('新增音符需要起点、尾端和轨道')
        new = replace(old, **changes) if old else Note(**changes, source='manual',id='manual:'+secrets.token_hex(10))
        note_from(new.data())  # validate type/range before any snapping
        if not strict:
            if duration_ms is not None and new.end > duration_ms:
                raise ValueError('音符不能超出音频结尾')
            snapped,_ = align_candidates([new], variant, confirmed=True)
            if new.hold and not snapped[0].hold:
                raise ValueError('LN 在当前细分下不足一格，请调整尾端或改为短键')
            new = replace(snapped[0],raw='')
            if duration_ms is not None and new.end > duration_ms:
                raise ValueError('量化后的尾端超出音频结尾，请缩短一格')
        result = [new if n is old else n for n in notes] if old else notes+[new]
    if not result:
        raise ValueError('请至少保留一个音符')
    report = validate(result, variant['settings']['keys'])
    if not report['valid']:
        raise ValueError('修改未保存：'+report['issues'][0])
    result.sort(key=lambda n:(n.start,n.lane,n.end))
    return result, dict(op=op,before=old.data() if old else None,after=new.data() if new else None)
