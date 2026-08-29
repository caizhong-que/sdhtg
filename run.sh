#!/bin/bash
# 保存为 run_all_structs.sh，运行前赋予执行权限：chmod +x run_all_structs.sh

# 定义种子数组
seeds=(42 123 256 512)

# 定义所有结构变体（包括 struct_base 和其余 6 个）
structures=(
    struct_base
    struct_film
    struct_action
    struct_entity
    struct_temporal
    struct_semantic
    struct_full
)

# 双重循环：外层种子，内层结构
for seed in "${seeds[@]}"; do
    for struct in "${structures[@]}"; do
        echo "Running: seed=$seed, struct=$struct"
        python scripts/train.py \
            --config configs/experiment/bgl.yaml \
            --model-config "configs/model/${struct}.yaml" \
            --skip-pretrain \
            --tag "$struct" \
            --seed "$seed"
        # 检查返回码，如果失败则退出（可选）
        if [ $? -ne 0 ]; then
            echo "Error: training failed for seed=$seed, struct=$struct"
            exit 1
        fi
    done
done

echo "All experiments completed."