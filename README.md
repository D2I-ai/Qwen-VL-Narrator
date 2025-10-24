# Model Card for Qwen-VL-Narrator

<p>
  <a href="README-CN.md" target="_blank">
    <img src="https://img.shields.io/badge/%E4%B8%AD%E6%96%87%E7%AE%80%E4%BB%8B-purple?style=for-the-badge&logo=markdown" alt="Chinese README"/>
  </a>
  <a href="https://huggingface.co/xiaosu-zhu/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge" alt="HuggingFace Model"/>
  </a>
  <a href="https://modelscope.cn/models/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/ModelScope-red?color=1978fe&style=for-the-badge" alt="ModelScope Model"/>
  </a>
  <a href="https://huggingface.co/spaces/xiaosu-zhu/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fspaces%2Fxiaosu-zhu%2FQwen-VL-Narrator&query=%24.runtime.stage&style=for-the-badge&logo=huggingface&label=Demo&color=yellow" alt="Demo"/>
  </a>
  <a href="https://modelscope.cn/studios/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator" target="_blank">
    <img src="https://img.shields.io/badge/Demo-red?style=for-the-badge&color=33cc33&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB2ZXJzaW9uPSIxLjEiIHdpZHRoPSIyNCIgaGVpZ2h0PSIxNCIgdmlld0JveD0iMCAwIDI0IDE0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgo8dGl0bGU%2BTW9kZWxTY29wZSBCYWRnZTwvdGl0bGU%2BCjxnIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI%2BCjxnIGZpbGwtcnVsZT0ibm9uemVybyI%2BCjxwYXRoIGQ9Im0wIDIuNjY3aDIuNjY3djIuNjY3aC0yLjY2N3YtMi42Njd6bTggMi42NjZoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3oiIGZpbGw9IiMzNkNFRDAiLz4KPHBhdGggZD0ibTAgNS4zMzNoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3ptMi42NjcgMi42NjdoMi42NjZ2Mi42NjdoMi42Njd2Mi42NjZoLTUuMzMzdi01LjMzM3ptMC04aDUuMzMzdjIuNjY3aC0yLjY2N3YyLjY2NmgtMi42NjZ2LTUuMzMzem04IDhoMi42Njd2Mi42NjdoLTIuNjY3di0yLjY2N3oiIGZpbGw9IiM2MjRBRkYiLz4KPHBhdGggZD0ibTI0IDIuNjY3aC0yLjY2N3YyLjY2N2gyLjY2N3YtMi42Njd6bS04IDIuNjY2aC0yLjY2N3YyLjY2N2gyLjY2N3YtMi42Njd6IiBmaWxsPSIjMzZDRUQwIi8%2BCjxwYXRoIGQ9Im0yNCA1LjMzM2gtMi42Njd2Mi42NjdoMi42Njd2LTIuNjY3em0tMi42NjcgMi42NjdoLTIuNjY2djIuNjY3aC0yLjY2N3YyLjY2Nmg1LjMzM3YtNS4zMzN6bTAtOGgtNS4zMzN2Mi42NjdoMi42Njd2Mi42NjZoMi42NjZ2LTUuMzMzeiIgZmlsbD0iIzYyNEFGRiIvPgo8L2c%2BCjwvZz4KPC9zdmc%2BCg%3D%3D" alt="Demo"/>
  </a>
</p>

**[[中文版本]](README-CN.md)**

**Qwen-VL-Narrator** is an expert model for understanding video clips from film and TV dramas and designed to generate fine-grained descriptions of characters, scenes, and filming techniques. It can be applied to scenarios such as video retrieval, summarization, understanding, and fine-grained annotation. It also helps for video generation by producing accurate prompt to "clone" a video.

Please try:

* [HuggingFace Demo](https://huggingface.co/spaces/xiaosu-zhu/Qwen-VL-Narrator) 
* [ModelScope Demo](https://modelscope.cn/studios/Apsara_Lab_Multimodal_Intelligence/Qwen-VL-Narrator) (Chinese only)

#### Highlights

* **Small Model Size**: The model is fine-tuned from Qwen2.5-VL 7B, allowing for easy deployment on a single H20, L20, or even a 5090 GPU.
* **High-Quality Video Descriptions**: Thanks to the diversity of its training samples, the model can provide more detailed video descriptions than previous models, demonstrating excellent performance for precise and comprehensive annotation.
* **Integration with Workflows**: The model can be integrated into film and television production workflows to provide summary information for video clips to other modules, enabling capabilities like long-video consolidation and structured output.

## 📖 Model Capabilities

Qwen-VL-Narrator's core capabilities include:

* **👥 Character Understanding**: The model can describe the appearance and demeanor of characters in detail, including but not limited to: facial features, body shape, clothing, actions, and expressions.
* **🏞️ Scene Understanding**: The model can describe the environment and setting in detail, including but not limited to: location, lighting, props, and atmosphere.
* **📝 Story Telling**: The model can describe events and actions in the video in detail, and can describe character dialogues based on subtitles.
* **🎬 Technical Analysis**: The model can analyze professional filmmaking techniques in detail, including but not limited to: camera movement, composition, color, staging, and transitions.

---

## 🌟 Showcase

Video | Description
--- | ---
![](https://github.com/D2I-ai/Qwen-VL-Narrator/blob/main/assets/demo.gif?raw=true) | This video clip shows a dialogue scene taking place at night. The main subject is a young man, facing sideways to the camera, talking to a woman whose back is partially visible on the left side of the frame. The man has sharp facial features, dark hair tied in a bun at the back of his head, and is wearing a dark blue or black collared top. The scene is set in a high place overlooking a city at night. The background features blurred city lights, creating colorful light spots (bokeh effect), which establishes a quiet and slightly detached urban night atmosphere. A thin purple line (resembling a decoration or cable) hangs vertically from the top of the frame, near the man's head.<br/>At the beginning of the video, the man is turned towards the woman, his gaze focused and his expression calm. As the dialogue unfolds (indicated by the subtitles), his expression changes subtly. When the subtitle "What" appears, his lips move slightly, and his eyes show a questioning look. Then, the subtitle "Ah, it's nothing" appears; his expression becomes slightly hesitant and avoidant, and his gaze briefly shifts downward. Next, the subtitles "You say," "I didn't even finish high school," and "I don't know how to do anything" appear in sequence. During this time, his expression turns to clear frustration and self-deprecation, with his brow furrowed and his eyes downcast, revealing vulnerability and unease. Finally, with the subtitles "So what" and "So do you think I can still be saved," his emotion deepens. His facial muscles tighten, his eyes are filled with pleading and helplessness, and eventually, his gaze drops completely, showing deep self-doubt.<br/>From a technical visual perspective, the focus is precisely on the man's face, making his detailed expressions clearly visible, while the background city lights are blurred into light spots, emphasizing the main character. In terms of composition, the man occupies the main position on the right side of the frame, while the woman's back is on the left edge, creating a visual relationship of dialogue. In terms of color, the overall tone is dark, dominated by the deep blues and blacks of the night scene. The colorful light spots in the background (yellow, orange, white, red, etc.) contrast with the dark clothing of the foreground character, and the purple line adds a touch of brightness. Regarding lighting, the main light source comes from the front of the man, illuminating his profile and creating a soft contrast between light and shadow, which highlights his facial contours and emotional expression. Throughout the clip, the camera remains stable with no noticeable movement or cuts. Through the actor's performance and the progression of the subtitles, the dynamic process of the man's emotional breakdown from calmness is fully presented.


For more use cases, such as structured output, image description, etc., [**please use the API we provide**](mailto:zhuxiaosu.zxs@alibaba-inc.com).


---

## 🚀 How to Use

Minimal example for video inference.

> [!TIP]
> For better inference performance, please deploy the model via `vllm` or `sglang`.

```bash
pip install "transformers>=4.45.0" accelerate qwen-vl-utils[decord]
```

This model is fine-tuned from Qwen2.5-VL, and its usage is the same as the original model. To do a inference, please provide the video content and use the following prompt.

Recommended video length is within 1 minute. Recommended video parameters:

```python
{ "max_pixels": 784 * 441, "fps": 2.0, "max_frames": 96, "min_frames": 16 }
```

```python
# Code sample from https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

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
            {"type": "text", "text": r"""Requirements:
- A precise description of the characters' appearance, clothing, actions, and expressions, including an analysis of their race/skin color.
- A detailed analysis of the set design, atmosphere, props, and environment.
- An objective and accurate presentation of the video's plot and narrative (with inferences aided by subtitles).
- An analysis of filming techniques, including camera language, shot types, and focus.
- Artistic processing including emotions/intentions are prohibited. Output only objective descriptions.

Output Format: Integrate the above content into a single, natural, and fluent paragraph describing the video clip. The description must be logically coherent and clear."""},
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
            {"type": "text", "text": r"""Requirements:
- A precise description of the characters' appearance, clothing, actions, and expressions, including an analysis of their race/skin color.
- A detailed analysis of the set design, atmosphere, props, and environment.
- An objective and accurate presentation of the video's plot and narrative (with inferences aided by subtitles).
- An analysis of filming techniques, including camera language, shot types, and focus.
- Artistic processing including emotions/intentions are prohibited. Output only objective descriptions.

Output Format: Integrate the above content into a single, natural, and fluent paragraph describing the video clip. The description must be logically coherent and clear."""},
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
                "video": "https://url.for.a.video",
                "max_pixels": 784 * 441,
                "fps": 2.0,
                "max_frames": 96,
                "min_frames": 16
            },
            {"type": "text", "text": r"""Requirements:
- A precise description of the characters' appearance, clothing, actions, and expressions, including an analysis of their race/skin color.
- A detailed analysis of the set design, atmosphere, props, and environment.
- An objective and accurate presentation of the video's plot and narrative (with inferences aided by subtitles).
- An analysis of filming techniques, including camera language, shot types, and focus.
- Artistic processing including emotions/intentions are prohibited. Output only objective descriptions.

Output Format: Integrate the above content into a single, natural, and fluent paragraph describing the video clip. The description must be logically coherent and clear."""},
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

## 💡 Use Cases

Qwen-VL-Narrator can be applied in various domains to automate video analysis and support downstream applications.

*   **Content Indexing and Search**: Create detailed, searchable metadata for large video archives, making it easy for users to find specific scenes, characters, or shots.
*   **Pre-production and Scripting**: Analyze raw footage to quickly generate video summaries or production scripts.
*   **Automated Audio Description**: Automatically generate audio descriptions for visually impaired audiences, providing accessible content.
*   **Video Generation Data Annotation**: Provide video-text annotation data for video generation models, achieving high-quality video-text alignment and enhancing the instruction-following ability.

## Limitations and Bias

This model, like all large language and vision models, has limitations.

* Due to biases and quality issues in the training data, the model's output may not be completely accurate and may contain hallucinations.
* The quality of descriptions may vary depending on the video's type, style, and content complexity.
* Due to the architectural limitations of Qwen2.5-VL, the model cannot process or describe audio.
* When the input video duration exceeds 1 minute, the description quality may decline. Please segment and preprocess the video according to your workflow.


## Contributors

Xiaosu Zhu, Sijia Cai, Bing Deng and Jieping Ye ***@ Data to Intelligence Lab, Alibaba Cloud***

This model was made possible through the close collaboration in Data to Intelligence Lab, Alibaba Cloud.
