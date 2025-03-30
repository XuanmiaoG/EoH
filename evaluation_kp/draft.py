import numpy as np
import os


def read_and_describe_npy(dataset_name: str):
    """
    读取指定的 .npy 文件，并打印其形状与具体结构信息。
    同时输出当前工作目录以及文件的绝对路径。
    """
    # 当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作路径 (current working directory): {current_dir}")

    # 拼接文件路径（相对路径）
    dataset_path = os.path.join('evaluation_kp/dataset', dataset_name)
    print(f"文件相对路径: {dataset_path}")

    # 文件绝对路径
    abs_dataset_path = os.path.abspath(dataset_path)
    print(f"文件绝对路径: {abs_dataset_path}")

    # 加载 .npy 文件
    data = np.load(dataset_path)

    # 输出基本形状信息
    print(f"\n数据已加载：{dataset_path}")
    print(f"数据的整体形状：{data.shape}")
    print(f"数据的dtype：{data.dtype}")
    print(f"数据的维度数 (ndim)：{data.ndim}")

    # 可以选择是否打印“整个数据”，但如果数据很大就会刷屏
    # 你可以注释或取消注释下面这行：
    # print(f"\n完整数据预览：\n{data}")

    # 如果你只想看前几个元素，可以像这样：
    if data.ndim == 1:
        # 一维情况
        print(f"\n前5个元素：\n{data[:5]}")
    elif data.ndim == 2:
        # 二维情况
        print(f"\n前5行数据：\n{data[:5]}")
    elif data.ndim == 3:
        # 三维情况
        print(f"\n前5个实例的前3行数据：")
        for idx in range(min(5, data.shape[0])):
            print(f"\n-- 第 {idx} 个实例 --")
            print(data[idx][:3])
    else:
        print("\n数据维度超过3，示例切片暂未实现，请根据实际需求修改。")


if __name__ == "__main__":
    # 假设文件名为 test50_dataset.npy，也可以替换成其他名字
    read_and_describe_npy("test50_dataset.npy")
