import paddle

def check_paddle_installation():
    # 检查PaddlePaddle是否能正常导入
    print("1. PaddlePaddle导入测试...")
    print(f"Paddle版本: {paddle.__version__}")
    
    # 检查基础张量运算
    print("\n2. 基础张量运算测试...")
    try:
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = paddle.to_tensor([4.0, 5.0, 6.0])
        z = x + y
        print(f"张量加法结果: {z.numpy()}")
    except Exception as e:
        print(f"张量运算失败: {str(e)}")
        return False
    
    # 检查CUDA是否可用(如果安装的是GPU版本)
    print("\n3. GPU支持检查...")
    print(f"Paddle是否编译了CUDA支持: {paddle.is_compiled_with_cuda()}")
    print(f"当前设备是否可用GPU: {paddle.device.is_compiled_with_cuda() and paddle.device.get_device() != 'cpu'}")
    
    return True

if __name__ == "__main__":
    print("=== 开始PaddlePaddle安装验证 ===")
    if check_paddle_installation():
        print("\n=== 验证通过 ===")
        print("PaddlePaddle安装正确，基本功能正常")
    else:
        print("\n=== 验证失败 ===")
        print("PaddlePaddle安装可能存在问题")