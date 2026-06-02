# 仓库指南

## 项目结构与模块组织

本仓库实现 FPO++ 机械臂操作实验。核心 Python 模块位于 `src/`，包括 flow policy、网络结构、环境封装和工具函数。顶层入口脚本包括 `pretrain_flow_bc.py`、`finetune_online_rl.py`、`eval_checkpoint.py` 和 `plot_results.py`。实验启动脚本位于 `scripts/`，W&B sweep 配置位于 `scripts/sweeps/`，云端启动封装位于 `scripts/skypilot/`。复现实验说明见 `docs/reproduce.md`。`thirdparty/` 是 vendored 依赖，除非明确更新依赖，否则不要修改。

## 构建、测试与开发命令

运行实验前先创建并激活环境：

```bash
bash setup_env.sh
source source_env.sh
```

安装本地包元数据：

```bash
pip install -e .
```

预览或运行实验批次：

```bash
DRY_RUN=1 bash scripts/run_main_benchmark.sh
NUM_GPUS=1 bash scripts/run_main_benchmark.sh
```

常用直接入口包括 `python pretrain_flow_bc.py ...`、`torchrun --nproc_per_node=1 finetune_online_rl.py --distributed True ...` 和 `python eval_checkpoint.py ...`。

## 编码风格与命名约定

使用 Python 3.10+。遵循现有风格：4 空格缩进、基于 dataclass 的配置、在实用场景中添加类型标注，并通过 `tyro` 暴露显式 CLI 参数。模块名使用小写加下划线，例如 `flow_model_config.py`。实验名应包含方法、任务和必要的 seed。`pyproject.toml` 未配置项目级 formatter 或 linter；保持 import 有序，避免无关的大范围格式化。

## 测试指南

当前仓库没有一方测试；`thirdparty/` 下的测试属于 vendored 包。轻量校验可对修改过的一方文件运行语法检查：

```bash
python -m compileall src pretrain_flow_bc.py finetune_online_rl.py eval_checkpoint.py
```

行为变更应在 PR 描述中附上可复现的小命令，例如 dry run、短评估或 reduced-step 训练命令。

## Commit 与 Pull Request 指南

当前 checkout 没有可用 Git 历史，因此无法推断本地提交规范。提交信息使用简洁祈使句，例如 `Add checkpoint evaluation option` 或 `Fix FPO ratio logging`。PR 应包含修改动机、受影响脚本或模块、验证命令，以及复现实验所需的 W&B run ID 或 checkpoint 路径。修改 `plot_results.py` 输出时附上图表或截图。

## 安全与配置提示

不要提交 W&B token、本地 checkpoint、数据集或生成的实验输出。机器相关路径应放在本地 shell 配置或命令行参数中，不要硬编码到源码里。
