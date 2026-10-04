# 原始数据获取与准备

公开仓库保留正式结果与复算所需原型证据。原始医学数据、ICD标准表和图谱派生缓存由研究者按官方来源取得并在本地准备，不从旧私有备份恢复。

## CHIP-CDN和ICD标准表

官方入口：[CBLUE天池数据集](https://tianchi.aliyun.com/dataset/95414)，下载CHIP-CDN材料；可能需要按平台要求申请或学生认证。

保存到 `raw/chip-cdn/`：CHIP-CDN_train.json（6,000条）、CHIP-CDN_dev.json（2,000条）、CHIP-CDN_test.json（本次快照10,000条）和 `国际疾病分类 ICD-10北京临床版v601.xlsx`。本次标准表含40,474行编码、37,645个去重标准名称。

官网CBLUE页面标注主要采用CC BY-NC-SA 4.0，不同任务有其他协议时在对应章节说明；使用时保留CBLUE/CHIP-CDN来源与引用，遵守原始下载协议。许可证参考：[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。本项目公开的保存预测/原型证据属于CHIP-CDN实验衍生材料，按该非商业、署名、相同方式共享条件提供。

## CPubMed-KG 2.0

官方入口：[CPubMed-KG](https://cpubmed.openi.org.cn/graphwiki/kgI)。保存为 `raw/cpubmedkg/CPubMed-KGv2_0.txt`。

本次快照278,665,209字节、4,580,058条三元组、45种关系。原项目来源记录标注CC BY-ND 4.0；本公开版不重新镜像原始图谱或图谱派生缓存。按官方说明下载、署名并在本地处理，禁止演绎限制不能用仓库其他材料的许可代替。

## 文件校验和生成

五个原始文件的精确大小与SHA-256见 `原始数据校验清单.json`。在 `实验/程序` 建立环境并安装项目后运行：

```bash
python scripts/prepare_official_data.py --check-only
python scripts/prepare_official_data.py
```

默认原始目录为外层“数据”；脚本使用seed=2026、训练内部验证比例0.1，调用本项目原始prepare_phase1_data和build_kg_index实现。

之后运行build_official_retrieval_data_v3.py，再按程序README的正式链路生成训练材料。用户必须取得对应五个原始文件；仅下载代码或模型不能代替原始数据准备。
