# 2025.10.22 Wed
# matmul_add_relu
# Case2.1.2 relu(matmul(x, w) + b) 


import os
import subprocess
import numpy as np
import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

# os.environ["AP_WORKSPACE_DIR"] = "/daiwenhao/Paddle/ap_workspace"
CASE_NAME = "Case2.1.3 gelu(matmul(x, w) + b) "

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
    DType = pct.DTypeVar("T", "float16")

    def foo(
        x: pct.Tensor([B, M, K], DType),
        w: pct.Tensor([K, N], DType),
        b: pct.Tensor([B, M, N], DType),
    ):
        y = paddle.matmul(x, w) 
        tmp = y + b
        tmp2 = paddle.nn.functional.gelu(tmp) # 在此处修改为了 gelu，仅此而已
        
        return tmp2

    return foo

def main():
    dtype = 'float16'
    x_shape = [32, 16, 16]
    x = paddle.randn(x_shape, dtype=dtype)
    x.stop_gradient = False

    y_shape = [16, 16]
    y = paddle.randn(y_shape, dtype=dtype)
    y.stop_gradient = False

    b_shape = [32, 16, 16]
    b = paddle.randn(b_shape, dtype=dtype)
    b.stop_gradient = False

    foo = getSubGraph()
    
    fused_foo = pcc.compile(foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass")
    generated_pir_program = GetPirProgram(fused_foo, [x, y, b])
    
    
    if 'pd_op.ap_variadic' not in generated_pir_program:
        print(">>> fail: fusion failed, excludes pd_op.ap_variadic !!!")
        return
    
    print("="*15+"  " + CASE_NAME + "  "+ "="*15)
    if IsCertainDevices():
        ap_outs = fused_foo(x, y, b).numpy()
        dy_outs = foo(x, y, b).numpy()
        
        try:
            np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-1)
            print("Pass")
        except AssertionError:
            print("Fail: output mismatch")
    else:
        print("pass: device not supported or check skipped")

if __name__ == "__main__":
    main()