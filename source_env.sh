# source ./thirdparty/miniconda3/bin/activate fpo_manipulation
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# 创建 overall workspace
WORKSPACE_DIR=$SCRIPT_DIR/thirdparty
CONDA_ROOT=$WORKSPACE_DIR/miniconda3
ENV_ROOT=$CONDA_ROOT/envs/fpo_manipulation

eval "$($CONDA_ROOT/bin/conda shell.bash hook)"
conda activate $ENV_ROOT