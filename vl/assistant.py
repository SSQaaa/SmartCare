from .backend import create_vl_backend
from .prompts import build_vl_prompt
from .types import VLRequest, VLResponse


LOCATE_MODES = {"locate", "both"}


class VLAssistant:
    def __init__(
        self,
        backend: str = "qwen",
        model_name: str = "",
        fallback_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "",
    ):
        self.backend = create_vl_backend(
            backend=backend,
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            device=device,
        )

    def load(self):
        self.backend.load()

    def run(self, image, request: VLRequest) -> VLResponse:
        response_mode = (request.response_mode or "answer").strip().lower()
        intent = (request.intent or "describe_scene").strip() or "describe_scene"
        if intent not in {"describe_scene", "find_object"}:
            intent = "describe_scene"
        should_locate = request.need_box or response_mode in LOCATE_MODES
        boxes = []
        answer = ""
        raw_text = ""

        if should_locate:
            boxes = self.backend.ground(image, request.user_text, max_boxes=request.max_boxes)

        if response_mode in {"answer", "both"} or not should_locate:
            if intent == "describe_scene" and not request.user_text.strip():
                answer = self.backend.describe(image, max_new_tokens=request.max_new_tokens)
            else:
                prompt = build_vl_prompt(request)
                answer = self.backend.generate(image, prompt, max_new_tokens=request.max_new_tokens)
            raw_text = answer
        elif response_mode == "locate":
            if boxes:
                best = boxes[0]
                target = request.user_text.strip() or best.label or "target"
                answer = f"Found {target}: {best.box}"
            else:
                answer = f"Target not found: {request.user_text.strip() or 'target'}"
            raw_text = answer

        return VLResponse(
            intent=intent,
            response_mode=response_mode,
            answer=answer,
            boxes=boxes,
            raw_text=raw_text,
        )


def create_vl_assistant(
    backend: str = "qwen",
    model_name: str = "",
    fallback_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "",
) -> VLAssistant:
    return VLAssistant(
        backend=backend,
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        device=device,
    )
