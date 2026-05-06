from .types import VLRequest


def build_vl_prompt(request: VLRequest) -> str:
    intent = (request.intent or "describe_scene").strip().lower()
    user_text = request.user_text.strip() or "\u8bf7\u5206\u6790\u8fd9\u5f20\u56fe\u50cf\u3002"
    language_hint = "\u8bf7\u7528\u4e2d\u6587\u56de\u7b54\u3002" if request.language == "zh" else "Please answer in English."

    if intent == "find_object":
        task_prompt = (
            "\u8bf7\u6839\u636e\u7528\u6237\u63cf\u8ff0\u627e\u5230\u76ee\u6807\u7269\u54c1\uff0c"
            "\u5e76\u7b80\u8981\u8bf4\u660e\u5b83\u5728\u753b\u9762\u4e2d\u7684\u4f4d\u7f6e\u3002"
        )
    else:
        task_prompt = (
            "\u8bf7\u63cf\u8ff0\u8fd9\u5f20\u753b\u9762\uff0c\u91cd\u70b9\u8bf4\u660e\u4e3b\u8981\u7269\u4f53\u3001"
            "\u989c\u8272\u3001\u4f4d\u7f6e\u5173\u7cfb\u548c\u573a\u666f\u3002"
        )

    return (
        f"{task_prompt}\n"
        f"\u7528\u6237\u539f\u8bdd\uff1a{user_text}\n"
        f"\u610f\u56fe\uff1a{intent}\n"
        f"\u8f93\u51fa\u5f62\u5f0f\uff1a{request.response_mode}\n"
        f"{language_hint}\n"
        "\u5982\u679c\u753b\u9762\u4fe1\u606f\u4e0d\u8db3\uff0c\u8bf7\u660e\u786e\u8bf4\u4e0d\u786e\u5b9a\uff0c\u4e0d\u8981\u7f16\u9020\u3002\n"
        "\u56de\u7b54\u5c3d\u91cf\u7b80\u6d01\u3002"
    )
