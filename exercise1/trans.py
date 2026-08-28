# 补充所有必要的导入（核心修复点1）
import os
import gzip
import pickle
import pandas as pd
from pathlib import Path
from sklearn.utils import Bunch  # 兼容sklearn的Bunch对象

def pkz_to_csv(
        pkz_file_path: str,
        csv_save_path: str = "./pkz_converted.csv",  # 默认保存到当前目录
        encoding: str = "utf-8-sig",  # 解决Windows Excel打开中文乱码
        force_overwrite: bool = False  # 是否强制覆盖已有文件
) -> bool:
    """
    将 .pkz 文件转换为 CSV 并保存到本地，带完整的异常处理和路径控制

    参数:
        pkz_file_path: .pkz 文件的路径（绝对/相对）
        csv_save_path: CSV 保存路径（支持自定义目录+文件名）
        encoding: 文件编码（utf-8-sig 兼容Excel）
        force_overwrite: 若CSV已存在，是否强制覆盖（False则提示并退出）

    返回:
        bool: 转换成功返回True，失败返回False
    """
    # --------------------------
    # 步骤1：校验输入文件
    # --------------------------
    if not os.path.exists(pkz_file_path):
        print(f"❌ 错误：.pkz文件不存在 → {pkz_file_path}")
        return False

    if not pkz_file_path.endswith(".pkz"):
        print(f"⚠️ 警告：文件不是 .pkz 格式 → {pkz_file_path}")

    # --------------------------
    # 步骤2：校验输出路径（自动创建目录）
    # --------------------------
    csv_dir = os.path.dirname(csv_save_path)
    if csv_dir and not os.path.exists(csv_dir):  # 若目录不存在则创建
        Path(csv_dir).mkdir(parents=True, exist_ok=True)
        print(f"📁 自动创建保存目录 → {csv_dir}")

    # 检查文件是否已存在
    if os.path.exists(csv_save_path) and not force_overwrite:
        print(f"❌ 错误：CSV文件已存在，如需覆盖请设置 force_overwrite=True → {csv_save_path}")
        return False

    # --------------------------
    # 步骤3：加载 .pkz 数据
    # --------------------------
    try:
        with gzip.open(pkz_file_path, "rb") as f:
            pkz_data = pickle.load(f)
        print(f"✅ 成功加载 .pkz 文件 → {pkz_file_path}")
    except Exception as e:
        print(f"❌ 加载 .pkz 文件失败：{str(e)}")
        return False

    # --------------------------
    # 步骤4：数据解析（兼容更多数据类型，核心修复点2）
    # --------------------------
    try:
        # 兼容sklearn的Bunch对象（cal_housing_py3.pkz是sklearn的数据集，返回Bunch）
        if isinstance(pkz_data, Bunch):
            # 将Bunch对象转为字典，再转DataFrame
            bunch_dict = pkz_data.__dict__
            # 核心处理：cal_housing数据的核心是 data + target + feature_names
            if "data" in bunch_dict and "feature_names" in bunch_dict:
                df = pd.DataFrame(pkz_data.data, columns=pkz_data.feature_names)
                # 把target（房价中位数）作为一列加入
                if "target" in bunch_dict:
                    df["target"] = pkz_data.target
            else:
                # 通用Bunch转换：只保留可序列化的字段
                df = pd.DataFrame([{k: v for k, v in bunch_dict.items() if isinstance(v, (int, float, str, list, dict))}])
        elif isinstance(pkz_data, pd.DataFrame):
            df = pkz_data
        elif isinstance(pkz_data, list):
            # 列表套字典/列表套列表 都兼容
            df = pd.DataFrame(pkz_data)
        elif isinstance(pkz_data, dict):
            # 字典转DataFrame（按列组织）
            df = pd.DataFrame(pkz_data)
        else:
            # 非结构化数据：转为字典后再转DataFrame（适合单条记录）
            df = pd.DataFrame([pkz_data])
            print(f"⚠️ 提示：.pkz 为非结构化数据，已自动转为单条记录的表格")

        # 数据空值处理（可选，根据需求调整）
        df = df.fillna("")  # 空值替换为空字符串，避免CSV出现NaN
        print(f"✅ 数据解析完成 → 共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"❌ 数据解析失败（非结构化数据无法转为表格）：{str(e)}")
        return False

    # --------------------------
    # 步骤5：保存 CSV 到本地
    # --------------------------
    try:
        df.to_csv(
            csv_save_path,
            index=False,  # 不保存DataFrame索引
            encoding=encoding,  # 兼容中文
            sep=",",  # CSV分隔符（可改为\t生成TSV）
            quotechar='"',  # 处理含逗号的字段
            escapechar="\\"  # 转义特殊字符
        )
        # 获取文件绝对路径，方便用户查找
        abs_path = os.path.abspath(csv_save_path)
        print(f"🎉 CSV文件保存成功 → 绝对路径：{abs_path}")
        return True
    except Exception as e:
        print(f"❌ 保存 CSV 失败：{str(e)}")
        return False


# --------------------------
# 示例调用（直接运行即可）
# --------------------------
if __name__ == "__main__":
    # 示例1：默认保存到当前目录（核心修复点3：路径用原始字符串/双反斜杠）
    pkz_to_csv(
        # 方式1：原始字符串（推荐），避免转义问题
        pkz_file_path=r"E:\code\database\sklearn_data\cal_housing_py3.pkz",
        # 方式2：双反斜杠（等价）
        # pkz_file_path="E:\\code\\database\\sklearn_data\\cal_housing_py3.pkz",
        force_overwrite=True  # 允许覆盖已有文件
    )