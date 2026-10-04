# MEDNORM 正式程序

本目录对应CHIP-CDN官方训练6,000条、验证2,000条的最终方法，Micro-F1为71.7293%。

## 环境和数据

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

Python 3.11至3.13；数据下载来源和五个原始文件哈希见 `../数据/README.md`。在项目根目录运行 `python3 修复目录链接.py` 后，data/artifacts分别链接到外层数据/结果。

## 正式训练顺序

1. `scripts/prepare_official_data.py` 校验官方原始文件，构建基础数据与图谱索引。
2. `scripts/build_official_retrieval_data_v3.py` 构建召回数据。
3. `scripts/build_official_retriever_colab_bundle_v3.py` 打包，`colab/train_official_biencoder_v3.py` 训练召回模型。
4. `scripts/build_official_ranker_data_v3.py` 从正式召回输出构建精排、图谱画像与原型数据。
5. `scripts/build_official_components_colab_bundle_v3.py` 打包，`colab/train_official_count_v3.py`、`colab/train_official_ranker_v3.py` 训练数量和精排组件。
6. `scripts/build_official_atomic_colab_bundle_v3.py` 打包结构输入，`colab/generate_atomic_mentions_v3.py` 执行Qwen结构推理。
7. `scripts/evaluate_frozen_official_dev_fusion_v3.py` 融合，`scripts/validate_official_dev_result_v3.py` 独立复核。

Colab会话和授权需要自行建立。历史报告里的/content与gdrive路径需要映射到自己的运行环境。

## 保存结果复核

项目根目录的 `python3 检查资料包.py` 只使用标准库；在本目录也可运行 `.venv/bin/python -m mednorm.cli verify`。显式路径的融合复算命令见根目录 `03_复核与复现说明.md`。

正式13模型在GitHub Release，不依赖旧私有云盘会话。共享模块即使含v2或phase1名字，仍是正式实现的依赖；train_biencoder、train_count_v3、train_ranker_v3和export_fp16_checkpoint_v3被正式训练/打包程序调用，不能按名称视为旧实验。

学校Word文件保留可编辑成品；旧填表脚本和外部模板依赖已移除。论文构建器及其共享基类、字体和图表保留在论文目录。
