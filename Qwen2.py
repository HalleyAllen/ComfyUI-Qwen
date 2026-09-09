import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def _get_local_model_dirs():
    """动态解析本地模型存放目录，不写死任何路径。

    优先用 ComfyUI 运行时提供的 models 根目录（自动适配任意安装位置），
    取不到时再以插件所在位置反推（../.. 即 ComfyUI 根）作为兜底。
    """
    candidates = []
    try:
        from comfy import folder_paths
        base = getattr(folder_paths, "models_dir", None)
        if base:
            candidates.append(os.path.join(base, "LLM"))
    except Exception:
        pass
    candidates.append(
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "LLM")
        )
    )
    # 去重（同一路径可能同时命中两种来源）
    seen, dirs = set(), []
    for c in candidates:
        c = os.path.normpath(c)
        if c not in seen:
            seen.add(c)
            dirs.append(c)
    return dirs


# 下拉显示名 -> 真实路径 的映射（INPUT_TYPES 时刷新）
_LOCAL_MODEL_MAP = {}


def _scan_local_models():
    """扫描所有本地模型目录，返回 {显示名: 真实路径}（仅收录含 config.json 的 transformers 格式目录）"""
    global _LOCAL_MODEL_MAP
    mapping = {}
    for base in _get_local_model_dirs():
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                p = os.path.join(base, name)
                if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
                    label = f"{name}（本地）"
                    if label not in mapping:
                        mapping[label] = p
    _LOCAL_MODEL_MAP = mapping
    return mapping

class Qwen2_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        options = [
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-72B-Instruct",
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct",
        ]
        # 自动把本地已下载的 transformers 模型目录放到下拉框最前面（显示短名）
        local = list(_scan_local_models().keys())
        options = local + options
        return {
            "required": {
                "model_name": (options,),
            }
        }

    RETURN_TYPES = ("QWEN2", "TK")
    RETURN_NAMES = ("Qwen2", "tokenizer")
    FUNCTION = "load_model"
    CATEGORY = "⛱️Qwen2"
  
    def load_model(self, model_name):
        # 下拉框里选的是短名（如 "Qwen2.5-0.5B-Instruct（本地）"），还原成真实路径；
        # 同时也兼容旧工作流里保存的直接路径，以及 Hugging Face 仓库 ID。
        if model_name in _LOCAL_MODEL_MAP:
            model_name = _LOCAL_MODEL_MAP[model_name]
        if os.path.isdir(model_name):
            print(f"[Qwen2] Loading local model from: {model_name}")
        else:
            print(f"[Qwen2] Loading model from Hugging Face Hub: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


class Qwen2_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN2",),
                "tokenizer": ("TK",),
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "system_instruction": ("STRING", {"default": "You are creating a prompt for Stable Diffusion to generate an image. First step: understand the input and generate a text prompt for the input. Second step: only respond in English with the prompt itself in phrase, but embellish it as needed but keep it under 200 tokens.", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"

    CATEGORY = "⛱️Qwen2"


    def generate_content(self, model, tokenizer, prompt, system_instruction):

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return (response,)


class Qwen2_Chat_Zho:
    def __init__(self):
        self.chat_history = []

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN2",),
                "tokenizer": ("TK",),
                "prompt": ("STRING", {"default": "What is the meaning of life?", "multiline": True}),
                "system_instruction": ("STRING", {"default": "You are creating a prompt for Stable Diffusion to generate an image. First step: understand the input and generate a text prompt for the input. Second step: only respond in English with the prompt itself in phrase, but embellish it as needed but keep it under 200 tokens.", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_content"

    CATEGORY = "⛱️Qwen2"

    def qwen_2(self, user_question, system_role):
        messages = [{"role": "system", "content": system_role},
                    {"role": "user", "content": user_question}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    def generate_content(self, model, tokenizer, prompt, system_instruction):
        # Store model, tokenizer, and temperature as instance variables
        self.model = model
        self.tokenizer = tokenizer

        # Generate response and update chat history
        response = self.qwen_2(prompt, system_instruction)
        self.chat_history.append({"role": "user", "content": prompt})
        self.chat_history.append({"role": "system", "content": response})
        
        # Format and return chat history
        formatted_history = self.format_chat_history()
        return (formatted_history,)

    def format_chat_history(self):
        formatted_history = []
        for message in self.chat_history:
            formatted_message = f"{message['role']}: {message['content']}"
            formatted_history.append(formatted_message)
            formatted_history.append("-" * 40)  # Add a separator line
        return "\n".join(formatted_history)

        

NODE_CLASS_MAPPINGS = {
    "Qwen2_ModelLoader_Zho": Qwen2_ModelLoader_Zho,
    "Qwen2_Zho": Qwen2_Zho,
    "Qwen2_Chat_Zho": Qwen2_Chat_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2_ModelLoader_Zho": "⛱️Qwen2 ModelLoader",
    "Qwen2_Zho": "⛱️Qwen2",
    "Qwen2_Chat_Zho": "⛱️Qwen2 Chat",
}
