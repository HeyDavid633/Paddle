# 2025.10.29 Wed
# Attention_with_mask (matmul1 + mask + softmax + matmul2)
# Case1  Attention_with_mask


import os
import subprocess
import numpy as np
import paddle
import paddle.incubate.cc as pcc
import paddle.incubate.cc.typing as pct

CASE_NAME = "Case1 Attention_with_mask"

def paddle_attn_std(q, k, v, mask=None):
    # Q(B, H, S, W) @ K^T(B, H, W, S) -> (B, H, S, S)
    kt = k.transpose([0, 1, 3, 2])  # transpose(-2, -1)
    scores = paddle.matmul(q, kt)
    scores = scores / (q.shape[-1]  ** 0.5)
    
    # Apply mask
    scores = scores - 10000.0 * (1.0 - mask.astype(scores.dtype))
    
    # Softmax
    probs = F.softmax(scores, axis=-1)
    
    # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W)
    h = paddle.matmul(probs, v)
    return h

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
    B = pct.DimVar("B", 1)
    H = pct.DimVar("H", 12)
    S = pct.DimVar("S", 128)
    W = pct.DimVar("W", 64)
    DType = pct.DTypeVar("T", "float16")

    def foo(
        q: pct.Tensor([B, H, S, W], DType),
        k: pct.Tensor([B, H, S, W], DType),
        v: pct.Tensor([B, H, S, W], DType),
        mask: pct.Tensor([1, 1, S, S], DType),  # [1,1,S,S] for broadcasting
    ):
        # Step 1: q @ k^T and scale
        kt = paddle.transpose(k, [0, 1, 3, 2])  # [B, H, W, S]
        scores = paddle.matmul(q, kt)           # [B, H, S, S]
        # scale = (W.value ** -0.5)
        # scaled_scores = scores * scale
        scaled_scores = scores * 0.125 
        
        # Step 2: apply mask (elementwise)
        masked_scores = scaled_scores - 10000.0 * (1.0 - mask)
        
        # Step 3: softmax
        probs = paddle.nn.functional.softmax(masked_scores)  # [B, H, S, S]
        
        # Step 4: matmul2
        output = paddle.matmul(probs, v)  # [B, H, S, W]
        
        return output

    return foo

def main():
    dtype = 'float16'
    B, H, S, W = 1, 12, 128, 64

    q = paddle.randn([B, H, S, W], dtype=dtype)
    q.stop_gradient = False

    k = paddle.randn([B, H, S, W], dtype=dtype)
    k.stop_gradient = False

    v = paddle.randn([B, H, S, W], dtype=dtype)
    v.stop_gradient = False

    mask = paddle.tril(paddle.ones([1, 1, S, S], dtype=dtype))  # [1,1,S,S]
    mask.stop_gradient = True

    foo = getSubGraph()
    
    fused_foo = pcc.compile(foo, ap_path=f"{os.path.dirname(paddle.__file__)}/apy/matmul_pass")
    generated_pir_program = GetPirProgram(fused_foo, [q, k, v, mask])
    
    if 'pd_op.ap_variadic' not in generated_pir_program:
        print(">>> fail: fusion failed, excludes pd_op.ap_variadic !!!")
        print("Generated program:\n", generated_pir_program)
        return
    
    print("="*15 + "  " + CASE_NAME + "  " + "="*15)
    if IsCertainDevices():
        ap_outs = fused_foo(q, k, v, mask).numpy()
        dy_outs = foo(q, k, v, mask).numpy()
        
        try:
            np.testing.assert_allclose(dy_outs, ap_outs, atol=1e-1, rtol=1e-2)
            print("Pass")
        except AssertionError as e:
            print("Fail: output mismatch")
            print(e)
    else:
        print("Skip: device not A100, output check skipped")

if __name__ == "__main__":
    main()
