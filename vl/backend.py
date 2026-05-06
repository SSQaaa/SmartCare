from .object_finder_legacy import create_vl_finder, parse_user_query
from .types import VLBox


class VLBackend:
    def __init__(
        self,
        backend: str = "qwen",
        model_name: str = "",
        fallback_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "",
    ):
        self.backend = backend
        self.finder = create_vl_finder(
            backend=backend,
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            device=device,
        )

    def load(self):
        if hasattr(self.finder, "load"):
            self.finder.load()

    def generate(self, image, prompt: str, max_new_tokens: int = 128) -> str:
        if hasattr(self.finder, "_generate_text"):
            return self.finder._generate_text(image, prompt, max_new_tokens=max_new_tokens)
        if hasattr(self.finder, "describe_image"):
            return self.finder.describe_image(image, max_new_tokens=max_new_tokens)
        return "This backend does not support text generation."

    def describe(self, image, max_new_tokens: int = 128) -> str:
        if hasattr(self.finder, "describe_image"):
            return self.finder.describe_image(image, max_new_tokens=max_new_tokens)
        return self.generate(image, "Describe this image.", max_new_tokens=max_new_tokens)

    def ground(self, image, query_text: str, max_boxes: int = 5) -> list[VLBox]:
        query = parse_user_query(query_text)
        candidates = self.finder.find_target_box(image, query, max_boxes=max_boxes)
        return [
            VLBox(
                box=list(candidate.box),
                label=candidate.label,
                source=candidate.source,
            )
            for candidate in candidates
        ]


def create_vl_backend(
    backend: str = "qwen",
    model_name: str = "",
    fallback_model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
    device: str = "",
) -> VLBackend:
    return VLBackend(
        backend=backend,
        model_name=model_name,
        fallback_model_name=fallback_model_name,
        device=device,
    )
