"""① 数据准备：下载 ChnSentiCorp → 清洗 → 按约 8:2 切分训练集 / 测试集。

产出：
    data/raw/ChnSentiCorp.csv       原始数据（评论文本 + 0/1 标签，约 7700 条酒店评论）
    data/processed/train.csv        训练集（约 80%，供训练使用）
    data/processed/test.csv         测试集（约 20%，训练绝不接触，仅评估用）

用法：
    python prepare_data.py
"""

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# 数据源依次尝试：GitHub 原始文件 → jsDelivr CDN 镜像（国内直连更稳）
SOURCES = [
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
    "https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv",
]
TEST_RATIO = 0.2
SEED = 42


def download_raw() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "ChnSentiCorp.csv"
    if target.exists():
        print(f"[跳过] 原始数据已存在：{target}")
        return target
    last_err: Exception | None = None
    for url in SOURCES:
        try:
            print(f"[下载] {url}")
            urllib.request.urlretrieve(url, target)
            print(f"[完成] 已保存到 {target}")
            return target
        except Exception as err:  # 单个源失败就换下一个
            print(f"[失败] {type(err).__name__}: {err}")
            last_err = err
    raise SystemExit(f"所有下载源均失败，请检查网络/代理后重试：{last_err}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["review", "label"]).copy()
    df["review"] = df["review"].astype(str).str.strip()
    df = df[df["review"] != ""]
    df = df.drop_duplicates(subset=["review", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    pos = int((df["label"] == 1).sum())
    print(f"[清洗] {before} 条 → {len(df)} 条（去空值、去重）")
    print(f"[分布] 正面 {pos} 条 / 负面 {len(df) - pos} 条")
    return df


def main() -> None:
    raw_path = download_raw()
    df = clean(pd.read_csv(raw_path))

    train_df, test_df = train_test_split(
        df, test_size=TEST_RATIO, random_state=SEED, stratify=df["label"]
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[切分] 训练集 {len(train_df)} 条 → {train_path}")
    print(f"[切分] 测试集 {len(test_df)} 条（训练全程不接触）→ {test_path}")
    print("[完成] 数据就绪，可运行 train_lstm.py / train_bert.py")


if __name__ == "__main__":
    main()
