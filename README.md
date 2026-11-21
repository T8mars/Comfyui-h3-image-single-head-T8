# H3 Image Single Head · T8

MiniMax H3 Image 多步采样用 ComfyUI 自定义节点。将发布模型的六组 PDD 输出头在内存中替换为固定均值头，配合修正 LoRA 使用；不改写模型文件。

[模型下载](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy) · [推荐文生图](workflows/H3-Image-T8-T2I.json) · [图生图 / 编辑](workflows/H3-Image-T8-Edit.json) · [效果对照](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/blob/main/EXAMPLES.md)

> **50 步 direct MLP100 工作流需要本节点。** 如果导入后 `H3ImageSingleHead` 显示红叉，请先按下方步骤安装，再重启 ComfyUI。原生 6 步 CFG1 则不需要安装。

## 安装

在 ComfyUI 根目录执行：

```bash
git clone https://github.com/T8mars/Comfyui-h3-image-single-head-T8.git custom_nodes/Comfyui-h3-image-single-head-T8
```

重启 ComfyUI，搜索 **H3 Image 单头多步（实验）**，节点 ID 为 `H3ImageSingleHead`。也可[下载节点 ZIP](https://github.com/T8mars/Comfyui-h3-image-single-head-T8/archive/refs/heads/main.zip)，解压到 `custom_nodes`，保证 `__init__.py` 直接位于插件目录内。

- 仓库已包含约 2 MB 的 `mean_head.safetensors`，不要删除或单独移动。
- 需要支持 MiniMax H3 Image 和 `comfy_api.latest` 的 ComfyUI；依赖现有环境中的 `torch`、`safetensors`。
- 如果已经安装旧的 `h3_image_single_head`，请移出旧目录，避免同一节点重复加载。

## 使用

从 [Hugging Face 模型仓库](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy) 下载完整主干、direct MLP100 修正 LoRA、文本编码器与 VAE。第一次使用直接导入 [文生图 JSON](workflows/H3-Image-T8-T2I.json)；需要参考图时导入 [图生图 JSON](workflows/H3-Image-T8-Edit.json)，在 `LoadImage` 中选图（可用[配套示例](https://huggingface.co/t8star/Minimax-H3-Image-Exp-Comfy/resolve/main/inputs/prepared.png)）。

线上模型文件名带 `-T8`。如果已有不带后缀的本地旧文件，无需改名或重下，在工作流的加载器里手动选中原文件即可。

```text
UNETLoader → H3ImageSingleHead → LoraLoaderBypassModelOnly → CFGGuider
```

|项目|已测试配置|
|---|---|
|主干|`minimax_h3_image_full_bf16_comfy_qkvfix-T8.safetensors`|
|修正 LoRA|`experimental_direct_mlp100_rank16-T8.safetensors`|
|LoRA 强度|**+1**，使用旁路加载器|
|采样|Euler · 50 步 · CFG 3|
|Sigma|配套工作流内置 shift 3 的 51 点表，无需额外叠加 SHIFT 节点|

本节点必须放在 LoRA 之前，输入未经其它 patch 修改的主干。只验证过上述完整 BF16/FP32 主干；不能保证 pruned/量化模型兼容。

原生 **6 步 CFG1** 路线不使用本节点，也不加载这份修正 LoRA。

## 说明

- 节点只替换输出头，不修改主干训练权重；它本身不是“去 PDD”工具。均值头不是蒸馏前教师头。
- 配套 direct MLP100 在固定样本中改善皮肤、布料和毛发材质，但雨伞缺柄、局部编辑保真等问题仍存在。详见模型仓库的完整对照，不宣称 99% 原版恢复。
- 每次执行会校验头文件 SHA256、张量形状和精度；不自动下载文件、不访问在线 API。

头资产 SHA256：`fab15aa9a5e0e9d78bb85d3adb5e86b17f3d697b6e29e0fe8ac7f72715ad65b9`。

## 测试与许可

设置 `COMFYUI_PATH` 为 ComfyUI 根目录，使用该环境的 Python 执行 `python -m unittest discover -s tests -v`。测试在 CPU 上检查头资产、原生 ModelPatcher 替换数学和错误输入，不加载 66 GB 主干、不出图。

节点及头资产按随附 [LICENSE](LICENSE) 和 [NOTICE](NOTICE) 发布；头资产派生自 [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)。这是第三方实验节点，非 MiniMax 官方产品。

## T8star · 社媒与资源

- [B站](https://space.bilibili.com/385085361) · [YouTube](https://www.youtube.com/@T8star-Aix/) · [Hugging Face](https://huggingface.co/t8star)
- [API（贞贞的平价 AI 小屋）](https://www.zzai.app/sign-up?aff=5f4w) · [免费画廊](https://www.openzhenzhen.com) · [在线 AI 应用（RunningHub）](https://www.runninghub.ai/zh-cn/user-center/1907375370302308353/userPost?inviteCode=rh-v1121) · [ComfyUI 整合包](https://pan.quark.cn/s/264edb7e36bd)

API 和 RunningHub 为邀请链接，使用本地节点和工作流不需要注册。
