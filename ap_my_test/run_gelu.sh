export FLAGS_prim_enable_dynamic=true && export FLAGS_prim_all=true

# 打开 CINN 编译器相关 FLAG
export FLAGS_use_cinn=true
export FLAGS_group_schedule_tiling_first=true
# 打开 PIR模式
export FLAGS_enable_pir_api=true

export AP_WORKSPACE_DIR="/daiwenhao/Paddle/ap_workspace"

# 是否打印 Program IR信息
export FLAGS_print_ir=true

# 调试信息
export GLOG_v=0

export FLAGS_enable_ap=1

python matmul_add_gelu.py > log/gelu_add.log 2>&1