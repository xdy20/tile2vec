#!/usr/bin/env python
"""
Train Tile2Vec embeddings using pre-generated triplets CSV.

Example:
    /home/y/miniconda3/envs/geo/bin/python scripts/train_tile2vec.py \
        --city Shenzhen --epochs 10 --batch-size 64 --embedding-dim 128
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import torch
from torch import optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import triplet_csv_dataloader
from src.tilenet import make_tilenet
from src.training import train_triplet_epoch


DEFAULT_TRIPLETS = ROOT / "outputs" / "triplets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Tile2Vec model on Sentinel-2 triplets")
    parser.add_argument(
        "--city",
        choices=["Shenzhen", "Beijing"],
        help="城市名称，用于自动定位 triplets CSV",
    )
    parser.add_argument(
        "--triplets-csv",
        help="自定义 triplets CSV 路径（若提供则优先使用）",
    )
    parser.add_argument("--img-type", default="landsat", choices=["landsat", "naip", "rgb"])
    parser.add_argument("--bands", type=int, default=4, help="使用的波段数")
    parser.add_argument("--tile-size", type=int, default=104, help="训练前 resize 的像素大小")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=0.0, help="Triplet loss L2 正则系数")
    parser.add_argument("--n-triplets", type=int, default=None, help="训练时随机采样的三元组数量")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Tile2Vec 输出向量维度")
    parser.add_argument("--save-path", help="模型保存路径，默认放在 models/ 目录")
    parser.add_argument("--cpu", action="store_true", help="强制使用 CPU 训练")
    args = parser.parse_args()

    if not args.triplets_csv:
        if not args.city:
            parser.error("必须指定 --city 或 --triplets-csv 之一")
        triplets_path = DEFAULT_TRIPLETS / f"{args.city.lower()}_triplets.csv"
        if not triplets_path.exists():
            parser.error(f"未找到 {args.city} 的 triplets CSV: {triplets_path}")
        args.triplets_csv = str(triplets_path)

    if not args.save_path:
        models_dir = ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        city_tag = args.city.lower() if args.city else "custom"
        args.save_path = str(models_dir / f"tile2vec_{city_tag}.pth")
    else:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cuda = device.type == "cuda"
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 读取 triplets: {args.triplets_csv}")

    dataloader = triplet_csv_dataloader(
        args.triplets_csv,
        img_type=args.img_type,
        bands=args.bands,
        target_size=args.tile_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        n_triplets=args.n_triplets,
        augment=True,
    )

    model = make_tilenet(in_channels=args.bands, z_dim=args.embedding_dim)
    if cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        metrics = train_triplet_epoch(
            model,
            cuda,
            dataloader,
            optimizer,
            epoch,
            margin=args.margin,
            l2=args.l2,
            print_every=max(1000, args.batch_size * 10),
        )
        history.append(
            {
                "epoch": epoch,
                "loss": metrics[0],
                "l_n": metrics[1],
                "l_d": metrics[2],
                "l_nd": metrics[3],
            }
        )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(args),
        "history": history,
    }
    torch.save(checkpoint, args.save_path)
    print(f"[INFO] 模型已保存到 {args.save_path}")


if __name__ == "__main__":
    main()
