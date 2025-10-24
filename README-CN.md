# Qwen-VL-Narrator 模型简介

<p>
  <a href="README.md" target="_blank">
    <img src="https://img.shields.io/badge/README-purple?style=for-the-badge&logo=markdown" alt="English README"/>
  </a>
  <a href="https://huggingface.co/xiaosu-zhu/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/%E6%8A%B1%E6%8A%B1%E8%84%B8-yellow?style=for-the-badge" alt="HuggingFace Model"/>
  </a>
  <a href="https://modelscope.cn/models/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/%E9%AD%94%E6%90%AD%E7%A4%BE%E5%8C%BA-red?color=1978fe&style=for-the-badge" alt="ModelScope Model"/>
  </a>
  <a href="https://huggingface.co/spaces/xiaosu-zhu/Qwen-VL-Narrator-CN" target="_blank">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fspaces%2Fxiaosu-zhu%2FQwen-VL-Narrator-CN&query=%24.runtime.stage&style=for-the-badge&logo=huggingface&label=%E6%BC%94%E7%A4%BA&color=yellow" alt="Demo"/>
  </a>
  <a href="https://modelscope.cn/studios/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/%E6%BC%94%E7%A4%BA-red?style=for-the-badge&color=33cc33&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHdpZHRoPSIyNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDE0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8dGl0bGU%2BTW9kZWxTY29wZSBCYWRnZTwvdGl0bGU%2BCjxnIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI%2BCjxnIGZpbGwtcnVsZT0ibm9uemVybyI%2BCjxwYXRoIGQ9Im0wIDIuNjY3aDIuNjY3djIuNjY3aC0yLjY2N3YtMi42Njd6bTggMi42NjZoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3oiIGZpbGw9IiMzNkNFRDAiLz4KPHBhdGggZD0ibTAgNS4zMzNoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3ptMi42NjcgMi42NjdoMi42NjZ2Mi42NjdoMi42Njd2Mi42NjZoLTUuMzMzdi01LjMzM3ptMC04aDUuMzMzdjIuNjY3aC0yLjY2N3YyLjY2NmgtMi42NjZ2LTUuMzMzem04IDhoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3oiIGZpbGw9IiM2MjRBRkYiLz4KPHBhdGggZD0ibTI0IDIuNjY3aC0yLjY2N3YyLjY2N2gyLjY2N3YtMi42Njd6bS04IDIuNjY2aC0yLjY2N3YyLjY2N2gyLjY2N3YtMi42Njd6IiBmaWxsPSIjMzZDRUQwIi8%2BCjxwYXRoIGQ9Im0yNCA1LjMzM2gtMi42Njd2Mi42NjdoMi42Njd2LTIuNjY3em0tMi42NjcgMi42NjdoLTIuNjY2djIuNjY3aC0yLjY2N3YyLjY2Nmg1LjMzM3YtNS4zMzN6bTAtOGgtNS4zMzN2Mi42NjdoMi42Njd2Mi42NjZoMi42NjZ2LTUuMzMzeiIgZmlsbD0iIzYyNEFGRiIvPgo8L2c%2BCjwvZz4KPC9zdmc%2BCg%3D%3D" alt="Demo"/>
  </a>
</p>

**[[English Ver.]](README.md)**

**Qwen-VL-Narrator** 是一个影视领域专家模型，旨在为影视作品的短视频片段生成人物、场景和拍摄手法层面的细粒度描述。能够应用于视频检索、摘要、理解、细粒度标注等场景，也能用于视频生成工作流来实现视频反推。

请试用：
* [抱抱脸在线演示](https://huggingface.co/spaces/xiaosu-zhu/Qwen-VL-Narrator-CN)
* [魔搭社区在线演示](https://modelscope.cn/studios/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator)

#### 亮点

* **小尺寸参数量**：该模型基于 Qwen2.5-VL 7B 微调得来，能够方便地部署在单张 H20、L20 乃至 5090 上。
* **高质量视频描述**：得益于训练样本的多样性，该模型能够提供超越以往模型的详细视频描述，在描述全面性、准确性与通用性上，均具有优异表现。
* **易与工作流结合**：该模型可以接入影视制作工作流中，为其他流程提供视频片段总结摘要信息，从而实现长视频整合、结构化输出等能力

## 📖 模型能力

Qwen-VL-Narrator 核心能力包括：

* **👥 人物形象**：模型能够详细描述视频中人物的外貌与神态，包括但不限于：五官、体形、服饰、动作、表情等
* **🏞️ 场景理解**：模型能够详细描述视频中的环境与布景，包括但不限于：场所、光线、道具、氛围等
* **📝 叙事生成**：模型能够详细描述视频的事件与行为，并能根据字幕描述人物对话
* **🎬 技术手法**：模型能够详细解析专业影视制作手法，包括但不限于：运镜、构图、色彩、调度、转场等

---

## 🌟 案例展示

视频 | 描述
--- | ---
![](https://github.com/D2I-ai/Qwen-VL-Narrator/blob/main/assets/demo.gif?raw=true) | 该视频片段呈现了一场发生在夜间的对话场景。画面主体为一名年轻男性，他侧对着镜头，与画面左侧仅露出部分背影的女性进行交流。该男性具有清晰的面部轮廓，深色头发在脑后束成一个发髻，身着一件深蓝色或黑色的翻领上衣。场景设定在一个可以俯瞰城市夜景的高处，背景是虚化的都市灯火，呈现出多彩的光斑（散景效果），营造出一种静谧而略带疏离感的都市夜晚氛围。一根细长的紫色线条（类似装饰物或线缆）从画面顶部垂直悬挂下来，位于男性头部附近。<br/>视频开始时，男性侧脸对着女性，眼神专注，面部表情较为平静。随着对话展开（根据字幕提示），他的神态发生细微变化。当字幕显示“什么”时，他嘴唇微动，眼神中流露出询问。随后，字幕显示“啊 没什么”，他的表情略显迟疑和回避，视线有短暂的向下移动。接着，字幕依次出现“你说”、“我连高中都没念完”、“什么都不会”，在此期间，他的表情转为明显的沮丧和自我否定，眉头微蹙，眼神低垂，显露出脆弱和不安的情绪。最后，字幕显示“所以呢”、“所以你觉得我还有救吗”，他的情绪进一步深化，面部肌肉收紧，眼神中充满恳切与无助，最终视线完全垂下，流露出深深的自我怀疑。<br/>从视觉技术层面分析，焦点精确地落在男性面部，使得他的表情细节清晰可见，而背景的城市灯光则被处理成模糊的光斑，突出了人物主体。构图上，男性占据画面右侧主体位置，女性的背影位于左侧边缘，形成对话的视觉关系。色彩方面，整体色调偏暗，以夜景的深蓝、黑色为主，背景的彩色光斑（黄、橙、白、红等）与前景人物的深色衣着形成对比，紫色线条则增添了一抹亮色。打光方面，主要光源来自男性前方，照亮其侧脸，形成了较为柔和的明暗对比，突显了面部轮廓和情绪表达。整个片段中，镜头保持稳定，没有明显的移动或切换，通过人物表演和字幕内容的递进，完整地呈现了男性从平静到情绪逐渐崩溃的动态过程。


更多使用场景，如结构化输出、图片描述等，[**请使用我们提供的 API**](mailto:zhuxiaosu.zxs@alibaba-inc.com)


---

## 🚀 使用方法

用于视频推理的简单样例。

> [!TIP]
> 如有需要，请使用 `vllm` 或 `sglang` 部署推理服务，来获取更好的推理性能。

```bash
pip install "transformers>=4.45.0" accelerate qwen-vl-utils[decord]
```

本模型为 Qwen2.5-VL 微调模型，使用方式与原版模型无异，在调用时，提供视频内容并使用下列提示词即可。

推荐视频长度在 1 分钟以内，推荐视频参数：

```python
{ "max_pixels": 784 * 441, "fps": 2.0, "max_frames": 96, "min_frames": 16 }
```

```python
# 该示例来自 https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "xiaosu-zhu/Qwen-VL-Narrator", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("xiaosu-zhu/Qwen-VL-Narrator")


# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                    "max_pixels": 784 * 441,
                    "fps": [CALCULATED]
                ],
            },
            {"type": "text", "text": r"""要求：
- 对人物外貌、服饰、动作、神态的精确描述，并进行种族/肤色解析
- 对场景布置、氛围、道具、环境的详细解析
- 对视频情节、叙事的客观精准呈现（通过字幕辅助推理）
- 禁止进行艺术加工和情绪/意图的推断，仅提供客观描述
- 对镜头语言、景别、焦点等拍摄手法的分析

输出格式：整合上述内容，使用一段自然、流畅的中文对视频片段进行描述，逻辑通顺，条理清晰。"""},
        ],
    }
]

# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 784 * 441,
                "fps": 2.0,
                "max_frames": 96,
                "min_frames": 16
            },
            {"type": "text", "text": r"""要求：
- 对人物外貌、服饰、动作、神态的精确描述，并进行种族/肤色解析
- 对场景布置、氛围、道具、环境的详细解析
- 对视频情节、叙事的客观精准呈现（通过字幕辅助推理）
- 禁止进行艺术加工和情绪/意图的推断，仅提供客观描述
- 对镜头语言、景别、焦点等拍摄手法的分析

输出格式：整合上述内容，使用一段自然、流畅的中文对视频片段进行描述，逻辑通顺，条理清晰。"""},
        ],
    }
]

# Messages containing a video url and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://https://url.for.a.video",
                "max_pixels": 784 * 441,
                "fps": 2.0,
                "max_frames": 96,
                "min_frames": 16
            },
            {"type": "text", "text": r"""要求：
- 对人物外貌、服饰、动作、神态的精确描述，并进行种族/肤色解析
- 对场景布置、氛围、道具、环境的详细解析
- 对视频情节、叙事的客观精准呈现（通过字幕辅助推理）
- 禁止进行艺术加工和情绪/意图的推断，仅提供客观描述
- 对镜头语言、景别、焦点等拍摄手法的分析

输出格式：整合上述内容，使用一段自然、流畅的中文对视频片段进行描述，逻辑通顺，条理清晰。"""},
        ],
    }
]

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=1536)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

---

## 💡 应用场景

Qwen-VL-Narrator 可应用于不同领域，以实现视频分析自动化与下游应用。

* **内容检索**：为大型视频档案库创建详细、可搜索的元数据，方便用户查找特定场景、角色或镜头。
* **前期制作与脚本**：分析原始素材，快速生成视频摘要或影视制作脚本。
* **自动口述影像**：为视障观众自动生成口述影像，提供无障碍内容访问。
* **视频生成数据标注**：为视频生成模型提供视频-文本标注数据，实现高质量视频-文本对齐，提升模型指令遵循能力

## 模型偏见与局限性

与所有视觉-语言模型一样，本模型也存在局限性。
* 由于训练数据的偏好与质量问题，模型输出的描述并非完全准确，可能存在幻觉。
* 描述质量可能会随视频类型、风格和内容复杂性而有所差异。
* 由于 Qwen2.5-VL 架构的局限性，模型无法对音频进行描述。
* 当输入视频时长超过 1 分钟时，描述质量可能会出现下降。请基于你的工作流对视频进行切分预处理。


## Contributors

Xiaosu Zhu, Sijia Cai, Bing Deng and Jieping Ye ***@ Data to Intelligence Lab, Alibaba Cloud***
