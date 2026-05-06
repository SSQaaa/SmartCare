from dataclasses import dataclass, field


@dataclass
class VLBox:
    box: list[int]
    label: str = ""
    source: str = "vl"


@dataclass
class VLRequest:
    intent: str = "describe_scene"
    user_text: str = ""
    response_mode: str = "answer"
    need_box: bool = False
    language: str = "zh"
    max_boxes: int = 5
    max_new_tokens: int = 128


@dataclass
class VLResponse:
    intent: str
    response_mode: str
    answer: str = ""
    boxes: list[VLBox] = field(default_factory=list)
    raw_text: str = ""
