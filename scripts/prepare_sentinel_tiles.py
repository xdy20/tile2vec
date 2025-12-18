#!/usr/bin/env python
"""
Prepare Sentinel-2 tiles and Tile2Vec triplets for Shenzhen/Beijing.

Inputs
------
- data/Sentinel2/: 10 m Sentinel-2 mosaics or sub-tiles for each city.
- data/city_boundaries/Grid/{city}_grid.geojson: 1 km grid polygons.

Outputs (under baseline/tile2vec-master/outputs/)
-------------------------------------------------
- mosaics/{city}_sentinel2.tif
- tiles/{city}/{grid_id}.npy
- tiles/{city}_tiles_metadata.csv
- triplets/{city}_triplets.csv
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio import mask as rio_mask
from scipy.spatial import cKDTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Sentinel-2 tiles/triplets for Tile2Vec")
    parser.add_argument(
        "--cities",
        nargs="+",
        default=["Shenzhen", "Beijing"],
        choices=["Shenzhen", "Beijing"],
        help="要处理的城市列表",
    )
    parser.add_argument(
        "--neighbor-radius",
        type=float,
        default=3000.0,
        help="三元组中 neighbor 搜索半径（米）",
    )
    parser.add_argument(
        "--distant-radius",
        type=float,
        default=6000.0,
        help="三元组中 distant 的最小距离（米）",
    )
    parser.add_argument(
        "--triplets-per-anchor",
        type=int,
        default=2,
        help="每个 anchor 抽取的三元组数量",
    )
    parser.add_argument(
        "--max-triplets",
        type=int,
        default=50000,
        help="每个城市最多生成的三元组数量（0 表示不限制）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如 Mosaic/瓦片文件已存在，仍重新生成",
    )
    return parser.parse_args()


def city_file_map() -> Dict[str, Dict[str, Path]]:
    root = Path(__file__).resolve().parents[1]
    sentinel_dir = root / "data" / "Sentinel2"
    grid_dir = root / "data" / "city_boundaries" / "Grid"
    outputs_root = root / "outputs"
    return {
        "Shenzhen": {
            "raster_dir": sentinel_dir,
            "grid_path": grid_dir / "shenzhen_grid.geojson",
            "mosaic_path": outputs_root / "mosaics" / "shenzhen_sentinel2.tif",
            "tiles_dir": outputs_root / "tiles" / "shenzhen",
            "metadata_csv": outputs_root / "tiles" / "shenzhen_tiles_metadata.csv",
            "triplets_csv": outputs_root / "triplets" / "shenzhen_triplets.csv",
        },
        "Beijing": {
            "raster_dir": sentinel_dir,
            "grid_path": grid_dir / "beijing_grid.geojson",
            "mosaic_path": outputs_root / "mosaics" / "beijing_sentinel2.tif",
            "tiles_dir": outputs_root / "tiles" / "beijing",
            "metadata_csv": outputs_root / "tiles" / "beijing_tiles_metadata.csv",
            "triplets_csv": outputs_root / "triplets" / "beijing_triplets.csv",
        },
        "root": root,
    }


def find_city_rasters(city: str, raster_dir: Path) -> List[Path]:
    city_lower = city.lower()
    pattern = f"{city_lower}_sentinel2_10m*.tif"
    paths = sorted(raster_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"未找到 {city} 的 Sentinel-2 影像，期望匹配 {pattern}")
    return paths


def build_mosaic(paths: Sequence[Path], mosaic_path: Path, overwrite: bool):
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)
    if mosaic_path.exists() and not overwrite:
        print(f"[INFO] Mosaic 已存在，跳过: {mosaic_path}")
        return
    datasets = [rasterio.open(p) for p in paths]
    try:
        mosaic, transform = rio_merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="lzw",
        )
        with rasterio.open(mosaic_path, "w", **meta) as dst:
            dst.write(mosaic)
        print(f"[INFO] 写出 Mosaic: {mosaic_path}")
    finally:
        for ds in datasets:
            ds.close()


def pick_tile_id(row) -> str:
    for attr in ("id", "grid_id", "ID"):
        if hasattr(row, attr):
            value = getattr(row, attr)
            if value is not None and value != "":
                return str(value)
    return f"tile_{row.Index}"


def generate_tiles(mosaic_path: Path, grid_path: Path, tiles_dir: Path, metadata_csv: Path, overwrite: bool) -> pd.DataFrame:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mosaic_path) as src:
        grid = gpd.read_file(grid_path)
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)
        records = []
        for row in grid.itertuples():
            tile_id = pick_tile_id(row)
            tile_path = tiles_dir / f"{tile_id}.npy"
            if tile_path.exists() and not overwrite:
                centroid = row.geometry.centroid
                records.append(
                    {
                        "grid_id": tile_id,
                        "tile_path": str(tile_path),
                        "centroid_x": centroid.x,
                        "centroid_y": centroid.y,
                    }
                )
                continue
            geom = row.geometry
            try:
                out_image, _ = rio_mask.mask(src, [geom], crop=True)
            except ValueError:
                continue
            if out_image.size == 0:
                continue
            np.save(tile_path, out_image.astype(np.float32))
            centroid = geom.centroid
            records.append(
                {
                    "grid_id": tile_id,
                    "tile_path": str(tile_path),
                    "centroid_x": centroid.x,
                    "centroid_y": centroid.y,
                }
            )
    if not records:
        raise RuntimeError(f"{grid_path} 未生成任何瓦片，请检查影像是否覆盖")
    df = pd.DataFrame.from_records(records)
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(metadata_csv, index=False)
    print(f"[INFO] 保存瓦片元数据: {metadata_csv} (共 {len(df)} 条)")
    return df


def sample_triplets(
    df: pd.DataFrame,
    neighbor_radius: float,
    distant_radius: float,
    triplets_per_anchor: int,
    max_triplets: int,
) -> List[Tuple[str, str, str]]:
    coords = df[["centroid_x", "centroid_y"]].to_numpy()
    tile_paths = df["tile_path"].to_numpy()
    n = len(df)
    tree = cKDTree(coords)
    rng = np.random.default_rng(42)
    triplets: List[Tuple[str, str, str]] = []
    all_idx = np.arange(n)
    max_triplets = max_triplets if max_triplets > 0 else None

    for idx in range(n):
        neighbors = tree.query_ball_point(coords[idx], neighbor_radius)
        neighbors = [j for j in neighbors if j != idx]
        if not neighbors:
            continue
        nearby = tree.query_ball_point(coords[idx], distant_radius)
        mask = np.ones(n, dtype=bool)
        mask[nearby] = False
        mask[idx] = False
        distant_candidates = all_idx[mask]
        if distant_candidates.size == 0:
            continue
        for _ in range(triplets_per_anchor):
            neighbor_idx = int(rng.choice(neighbors))
            distant_idx = int(rng.choice(distant_candidates))
            triplets.append((tile_paths[idx], tile_paths[neighbor_idx], tile_paths[distant_idx]))
            if max_triplets and len(triplets) >= max_triplets:
                return triplets
    return triplets


def main():
    args = parse_args()
    files = city_file_map()
    outputs_root = files.pop("root") / "outputs"
    (outputs_root / "triplets").mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        cfg = files[city]
        print(f"\n===== 处理 {city} =====")
        raster_paths = find_city_rasters(city, cfg["raster_dir"])
        build_mosaic(raster_paths, cfg["mosaic_path"], overwrite=args.overwrite)
        df = generate_tiles(cfg["mosaic_path"], cfg["grid_path"], cfg["tiles_dir"], cfg["metadata_csv"], args.overwrite)
        triplets = sample_triplets(
            df,
            neighbor_radius=args.neighbor_radius,
            distant_radius=args.distant_radius,
            triplets_per_anchor=args.triplets_per_anchor,
            max_triplets=args.max_triplets,
        )
        if not triplets:
            raise RuntimeError(f"{city} 未生成任何三元组，请调整邻域/远距参数")
        triplets_df = pd.DataFrame(triplets, columns=["anchor", "neighbor", "distant"])
        triplets_df.to_csv(cfg["triplets_csv"], index=False)
        print(f"[INFO] 保存三元组: {cfg['triplets_csv']} (共 {len(triplets_df)} 条)")


if __name__ == "__main__":
    main()
