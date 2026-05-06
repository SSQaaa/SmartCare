from dataclasses import dataclass


@dataclass
class VoiceCommand:
    intent: str
    target: str = ""
    raw_text: str = ""


HELP_TEXT = "你可以说：看看周围、帮我找杯子、这是谁、这是什么等。"

SAFETY_CONFIRM_WORDS = (
    "没事",
    "我没事",
    "没关系",
    "我没关系",
    "我还好",
    "还好",
    "没问题",
    "不用担心",
    "安全",
    "没有摔倒",
)


def compact_text(text):
    return "".join(str(text or "").strip().lower().split())


def normalize_command_text(text):
    compact = compact_text(text)
    prefixes = (
        "你帮我",
        "请帮我",
        "帮我",
        "你给我",
        "给我",
        "请问",
        "请",
        "你",
        "我想",
        "我要",
        "俺想",
        "俺要",
        "俺",
        "我",
    )
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if compact.startswith(prefix):
                compact = compact[len(prefix):]
                changed = True
    fillers = ("一下", "一哈", "一下子", "吧", "呢", "啊", "呀", "可以", "能不能", "能否", "着")
    for filler in fillers:
        compact = compact.replace(filler, "")
    return compact


def has_any(text, words):
    return any(word in text for word in words)


def is_safety_confirmation(text):
    compact = normalize_command_text(text)
    return has_any(compact, SAFETY_CONFIRM_WORDS)


def strip_find_target(text):
    compact = normalize_command_text(text)
    prefixes = (
        "帮我找一下",
        "帮我找找",
        "帮我找",
        "找一下",
        "寻找",
        "查找",
        "找",
    )
    for prefix in prefixes:
        if compact.startswith(prefix):
            target = compact[len(prefix):]
            break
    else:
        target = compact

    possessives = ("我的", "我那个", "那个", "这个", "这只", "这本", "一个", "一只", "一本")
    changed = True
    while changed:
        changed = False
        for prefix in possessives:
            if target.startswith(prefix):
                target = target[len(prefix):]
                changed = True

    suffixes = ("在哪里", "在哪儿", "在哪", "的位置", "位置", "一下", "吧")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if target.endswith(suffix):
                target = target[:-len(suffix)]
                changed = True
    return target.strip()


def parse_command(text):
    raw_text = str(text or "").strip()
    compact = normalize_command_text(raw_text)
    if not compact:
        return VoiceCommand(intent="unknown", raw_text=raw_text)

    if compact in {"退出", "结束", "关闭", "拜拜", "再见"}:
        return VoiceCommand(intent="exit", raw_text=raw_text)

    if compact in {"帮助", "怎么用", "你会什么", "说明"}:
        return VoiceCommand(intent="help", raw_text=raw_text)

    if "停止当前功能" in compact or "停止功能" in compact or compact in {"停止", "停下"}:
        return VoiceCommand(intent="stop_current", raw_text=raw_text)

    if "跌倒状态" in compact or "摔倒状态" in compact:
        return VoiceCommand(intent="fall_status", raw_text=raw_text)

    if ("开始" in compact or "启动" in compact or "打开" in compact) and ("跌倒" in compact or "摔倒" in compact):
        return VoiceCommand(intent="start_fall_monitor", raw_text=raw_text)

    if ("停止" in compact or "关闭" in compact) and ("跌倒" in compact or "摔倒" in compact):
        return VoiceCommand(intent="stop_fall_monitor", raw_text=raw_text)

    if (
        compact in {"周围有什么", "检测物品", "识别物品", "附近有什么", "有什么东西", "看看周围有什么"}
        or (has_any(compact, ("周围", "附近", "旁边", "面前", "眼前")) and has_any(compact, ("有什么", "有啥", "东西", "物品")))
        or (has_any(compact, ("检测", "识别", "看看")) and has_any(compact, ("物品", "东西", "目标")))
    ):
        return VoiceCommand(intent="summarize_objects", raw_text=raw_text)

    if (
        compact in {"看看周围", "描述画面", "描述", "看周围", "场景描述", "看画面"}
        or (has_any(compact, ("描述", "说说", "看看", "看")) and has_any(compact, ("画面", "周围", "环境", "场景")))
        or (has_any(compact, ("告诉", "看看", "看", "识别", "描述", "说说")) and has_any(compact, ("这是什么", "这个是什么", "这东西", "这个东西", "这本书", "什么书", "这是什么书")))
        or compact in {"这是什么", "这个是什么", "这是什么书", "什么书"}
    ):
        return VoiceCommand(intent="describe_scene", raw_text=raw_text)

    if (
        compact in {"这是谁", "识别人脸", "看看是谁", "认人", "认一下人"}
        or (has_any(compact, ("谁", "人脸", "认人")) and has_any(compact, ("这", "识别", "看看", "看")))
    ):
        return VoiceCommand(intent="recognize_face", raw_text=raw_text)

    if compact.startswith(("帮我找", "找一下", "寻找", "查找", "找")) or has_any(compact, ("在哪里", "在哪", "找找")):
        target = strip_find_target(raw_text)
        return VoiceCommand(intent="find_object", target=target, raw_text=raw_text)

    return VoiceCommand(intent="unknown", raw_text=raw_text)
