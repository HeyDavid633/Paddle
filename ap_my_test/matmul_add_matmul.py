# 2025.10.28 Tue
# matmul_add_matmul
# Case2.2.1 matmul(matmul(a, b) + c), d)  入手案例
# 
# 麻烦的点在于，如何去捕捉最后一个matmul


import os
import subprocess
import numpy as np
import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

# os.environ["AP_WORKSPACE_DIR"] = "/daiwenhao/Paddle/ap_workspace"
CASE_NAME = "Case2.2.1 matmul(matmul(a, b) + c), d)"

def GetPirProgram(fused_func, tensor_args):
    dtypes = tuple(tensor.dtype for tensor in tensor_args)
    func = fused_func.func_overload_ctx.dtypes2func.get(dtypes, None)
    return str(func.infer_program.forward_program)

def IsCertainDevices():
    try:
        sp = subprocess.Popen(
            ['nvidia-smi', '-q'], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out_str = sp.communicate()[0].decode('utf-8')
        if 'A100' in out_str:
            return True
        else:
            return False
    except Exception as e:
        return False

def getSubGraph():
    B = pct.DimVar(32)
    M = pct.DimVar(16)
    K = pct.DimVar(16)
    N = pct.DimVar(16)
    H = pct.DimVar(16)
    DType = pct.DTypeVar("T", "float16")

    def foo(
        input0: pct.Tensor([B, M, K], DType),
        input1: pct.Tensor([K, N], DType),
        input2: pct.Tensor([B, M, N], DType),
        input3: pct.Tensor([B, N, H], DType),
    ):
        y = paddle.matmul(input0, input1) 
        tmp = y + input2
        result = paddle.matmul(tmp, input3)  #  matmul1 + add + matmul2
        
        return result

    return foo

def main():
    dtype = 'float16'
    input0_shape = [32, 16, 16]
    input0 = paddle.randn(input0_shape, dtype=dtype)
    input0.stop_gradient = False

    input1_shape = [16, 16]
    input1 = paddle.randn(input1_shape, dtype=dtype) 
    input1.stop_gradient = False

    input2_shape = [32, 16, 16]
    input2 = paddle.randn(input2_shape, dtype=dtype)
    input2.stop_gradient = False
    
    input3_shape = [32, 16, 16]
    input3 = paddle.randn(input3_shape, dtype=dtype)
    input3.stop_gradient = False
    
    foo = getSubGraph()
    
    fused_foo = pcc.compile(foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass")
    generated_pir_program = GetPirProgram(fused_foo, [input0, input1, input2, input3])
    
    
    if 'pd_op.ap_variadic' not in generated_pir_program:
        print(">>> fail: fusion failed, excludes pd_op.ap_variadic !!!")
        return
    
    print("="*15+"  " + CASE_NAME + "  "+ "="*15)
    if IsCertainDevices():
        ap_outs = fused_foo(input0, input1, input2, input3).numpy()
        dy_outs = foo(input0, input1, input2, input3).numpy()
        
        try:
            np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-1)
            print("Pass")
        except AssertionError:
            print("Fail: output mismatch")
    else:
        print("pass: device not supported or check skipped")

if __name__ == "__main__":
    main()