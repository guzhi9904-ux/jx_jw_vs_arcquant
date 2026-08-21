"""ARCQuant 离线校准/预处理入口。

输出的 ``reorder_index`` 与 ``select_num`` 是推理前的必需元数据：前者把
重要通道排到末尾，后者决定为每个线性层追加多少个残差通道（ARC）。
"""

from datasets import load_dataset
import torch.nn as nn
import gc
from utilize import * 
import torch
from collections import defaultdict
import functools
from typing import List
import time
from pathlib import Path
import pandas as pd
import numpy as np
import tqdm
import argparse
import math
import os
import time


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="Hugging Face 模型名或本地 checkpoint 路径。")
parser.add_argument(
    "--dataset", type=str, default="wikitext2", choices=["wikitext2", "c4", "humaneval", "pile"], 
    help="校准数据集；名称也会写入缓存文件名。"
)
parser.add_argument(
    "--act_sort_metric", type=str,
    help="通道重要性统计方式；当前实现中 mean/frobenius/max 最终都走逐通道绝对最大值。",
)
parser.add_argument("--samples", type=int, default=128, help="第一阶段统计激活的校准样本数。")
parser.add_argument("--seqlen", type=int, default=2048, help="每条校准样本的 token 长度。")


parser.add_argument(
    "--select-samples",
    type=int,
    default=32,
    help="Samples used by ARCQuant's select_num search stage.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda:0")
parser.add_argument(
    "--saved-dir",
    default="./saved",
    help="Directory for calibration artifacts (legacy default: ./saved).",
)
parser.add_argument(
    "--wikitext-cache-dir",
    default=os.environ.get("ARCQUANT_WIKITEXT_CACHE_DIR"),
    help="Optional directory containing wikitext-train.arrow/test.arrow.",
)

args = parser.parse_args()


DATASET_LOADERS = {
    "wikitext2": get_wikitext2,
    "c4": get_c4,
    "pile": get_pile,
    "humaneval": get_humaneval
}
        
def main():
    """依次完成激活统计、通道排序和每层 ARC 宽度估计。"""
    if args.wikitext_cache_dir:
        os.environ["ARCQUANT_WIKITEXT_CACHE_DIR"] = args.wikitext_cache_dir
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, enc = load_model(args.model)
    folder_path = Path(args.saved_dir)
    # Path(...).name works for the native Windows path passed by the local
    # reproduction script; splitting only on '/' treated the whole E:\\...
    # path as a filename and made torch.save fail after calibration.
    model_name = Path(args.model.rstrip('/\\')).name
    folder_path.mkdir(parents=True, exist_ok=True)

    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    start_time = time.time()
    
    print(f"Using {args.dataset} dataset for calibration.")
    get_dataset = DATASET_LOADERS[args.dataset]

    dataset_name = args.dataset.lower()
    act_scales_filename = folder_path / (
        f'{model_name.lower()}_act_scales_{dataset_name}_{args.act_sort_metric}.pt'
    )

    # 阶段 1：用 forward hook 收集每个 Linear 输入通道的统计量。
    print("Getting activation stats...")
    if not os.path.exists(act_scales_filename):
        print("Generating activation stats...")
        dataloader, _ = get_dataset(
            nsamples=args.samples,
            seed=args.seed,
            seqlen=args.seqlen,
            tokenizer=enc,
        )

        act_scales = get_act_stats(
            model,
            dataloader,
            args.device,
            metric=args.act_sort_metric,
            seqlen=args.seqlen,
        )
        torch.save(act_scales, act_scales_filename)
        del dataloader
    else:
        print("Loading pre-saved activation stats...")
        act_scales = torch.load(act_scales_filename, map_location="cpu")
        

    # 阶段 2：按统计量升序排列；最重要的通道因此集中在 index 尾部。
    print("Getting reording index...")
    reorder_index = get_reorder_index(model, act_scales, metric=args.act_sort_metric)
    
    # 阶段 3：另取 32 条样本估计各层需要补偿的通道数，并向上对齐到 64。
    print("Getting proportions...")

    _, inps = get_dataset(
                nsamples=args.select_samples,
                seed=args.seed,
                tokenizer=enc,
                seqlen=args.seqlen,
            )
    select_num, average_bits = search_select_proportions(
        model, inps, args.device, args.seqlen, reorder_index
    )
    
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # 量化评估阶段 model/main.py 会按完全相同的命名规则加载这些文件。
    reorder_filename = folder_path / (
        f'{model_name.lower()}_reorder_index_{dataset_name}_{args.act_sort_metric}.pt'
    )
    select_num_filename = folder_path / (
        f'{model_name.lower()}_select_num_{dataset_name}_{args.act_sort_metric}.pt'
    )
    avg_bits_filename = folder_path / (
        f'{model_name.lower()}_average_bits_{dataset_name}_{args.act_sort_metric}.pt'
    )

    print(f"Saving reorder index to {reorder_filename}")
    torch.save(reorder_index, reorder_filename)
    print(f"Saving select num to {select_num_filename}")
    torch.save(select_num, select_num_filename)
    print(f"Saving average bits to {avg_bits_filename}")
    torch.save(average_bits, avg_bits_filename)
    
if __name__ == "__main__":
    main()
