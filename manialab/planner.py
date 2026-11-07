"""Optional API planner: only bounded preferences; never creates note timestamps."""
import json
import urllib.error
import urllib.parse
import urllib.request

from .arrange import parameters

ALLOWED = {"density", "ln_ratio", "min_ln", "max_held", "chord_size", "focus", "pitch_follow", "balance", "jack_avoid", "release_gap",
           "phrase_ln", "drums_weight", "bass_weight", "vocals_weight", "other_weight"}


def validate_plan(value, segments):
    if not isinstance(value, dict) or set(value) - {"explanation", "preferences", "segments"}:
        raise ValueError("API 必须返回 explanation/preferences/segments JSON")
    prefs = value.get("preferences", {})
    if not isinstance(prefs, dict) or set(prefs) - ALLOWED:
        raise ValueError("API 返回了未允许的全局参数")
    parameters(prefs)
    result = []
    entries = value.get("segments", [])
    if not isinstance(entries, list) or len(entries) > len(segments):
        raise ValueError("API 返回的段落数无效")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"index", "preferences"}:
            raise ValueError("API 段落格式无效")
        index, local = entry["index"], entry["preferences"]
        if type(index) is not int or index < 0 or index >= len(segments) or index in seen:
            raise ValueError("API 段落索引无效或重复")
        if not isinstance(local, dict) or set(local) - ALLOWED:
            raise ValueError("API 返回了未允许的段落参数")
        parameters(local)
        seen.add(index)
        result.append({"start_ms": segments[index]["start_ms"], "end_ms": segments[index]["end_ms"], "preferences": local})
    explanation = value.get("explanation", "")
    if not isinstance(explanation, str):
        raise ValueError("API explanation 必须是文本")
    return {"preferences": prefs, "segments": result, "explanation": explanation[:3000]}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("API 地址返回重定向；请填写最终接口地址")


def plan(endpoint, model, key, instruction, segments, current, review=None):
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
        raise ValueError("API 地址格式无效")
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")):
        raise ValueError("API 地址必须为 HTTPS（本地服务可用 HTTP）")
    if not model.strip() or not instruction.strip():
        raise ValueError("请填写模型名和编排要求")
    prompt = ("你是 osu!mania 6–10K 编排参数助手。你没有听到音频，只能参考下列检测统计。"
              "不能声称听到了具体乐器，不允许创造音符、时间或改键数。返回 JSON 对象，不要 Markdown。"
              "结构: {explanation:中文说明,preferences:{参数},segments:[{index:已知段落索引,preferences:{参数}}]}。"
              "允许参数: density 1..24, ln_ratio 0..1, min_ln 100..2000毫秒, max_held 1..10, chord_size 1..10,"
              "focus balanced/pitched/transient, pitch_follow/balance/jack_avoid 0..1, release_gap 0..250毫秒。"
              "phrase_ln 0..1 控制有连续人声证据的跨音高长条；drums_weight/bass_weight/vocals_weight/other_weight 0..2 控制四声部取舍（0完全不采，1默认）。"
              "只有提供voices统计的段落才分析过分离声部，否则声部权重不起作用；不能把模型估计当作真实乐器标注。"
              "如有已有候选的review，它定位短放键、快速同轨和同手LN配合密集位置；结合用户目标调整相应段落，不能把这些提示都当错误。"
              "ln_ratio是目标LN物件数占比；程序在持续音证据和占轨约束下尽量达到，不能通过删短键或虚构音尾凑比例。不够证据时应在说明中指出。")
    payload = {"model": model.strip(), "messages": [{"role": "system", "content": prompt},
               {"role": "user", "content": json.dumps({"request": instruction[:6000], "current": current, "segments": segments, "review": review}, ensure_ascii=False)}],
               "temperature": .3, "max_tokens": 2500}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    url = endpoint.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=90) as response:
            body = response.read(128 * 1024 + 1)
        if len(body) > 128 * 1024:
            raise ValueError("API 响应过大")
        result = json.loads(body)["choices"][0]["message"]["content"].strip()
        if result.startswith("```"):
            result = result.split("\n", 1)[1].rsplit("```", 1)[0]
        return validate_plan(json.loads(result), segments)
    except urllib.error.HTTPError as error:
        raise ValueError(f"API 请求失败（HTTP {error.code}）；检查地址、模型、额度和密钥") from None
    except (urllib.error.URLError, TimeoutError):
        raise ValueError("API 连接失败或超时") from None
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        raise ValueError("API 未返回可用的 JSON 编排参数") from None
