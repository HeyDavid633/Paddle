# 2025.10.29 Wed
# Attention_with_mask (matmul1 + mask + softmax + matmul2)

import paddle
import paddle.nn.functional as F

def paddle_attn_std(q, k, v, mask=None):
    """
    Standard scaled dot-product attention in PaddlePaddle.
    
    Args:
        q: [B, H, S, W] - query
        k: [B, H, S, W] - key
        v: [B, H, S, W] - value
        mask: [B, S, S] or [S, S] - optional attention mask (1 for valid, 0 for masked)
    
    Returns:
        h: [B, H, S, W] - output of attention
    """
    # Q(B, H, S, W) @ K^T(B, H, W, S) -> (B, H, S, S)
    kt = k.transpose([0, 1, 3, 2])  # transpose(-2, -1) in PyTorch
    scores = paddle.matmul(q, kt)
    
    d_k = q.shape[-1] 
    scores = scores / (d_k ** 0.5)
    

    if mask is not None:
        if mask.ndim == 2:  # [S, S] -> [1, 1, S, S]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3: # [B, S, S] -> [B, 1, S, S]
            mask = mask.unsqueeze(1)
        scores = scores - 10000.0 * (1.0 - mask.astype(scores.dtype))
    probs = F.softmax(scores, axis=-1)
    
    # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W)
    h = paddle.matmul(probs, v)
    return h

if __name__ == "__main__":
    B, H, S, W = 1, 12, 128, 64
    
    q = paddle.randn([B, H, S, W])
    k = paddle.randn([B, H, S, W])
    v = paddle.randn([B, H, S, W])
    
    # mask = None    
    mask = paddle.tril(paddle.ones([S, S], dtype=paddle.float32))  # [S, S]
    
    # 前向计算
    output = paddle_attn_std(q, k, v, mask=mask)
    
    print("Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    print(f"Output shape: {output.shape}")  # [1, 12, 128, 64]