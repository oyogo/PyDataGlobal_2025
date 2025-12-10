import xarray as xr
import rioxarray
import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio.features import rasterize
from shapely.geometry import box
from dask.distributed import Client
import dask.array as da
from dask.diagnostics import ProgressBar
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import rasterio
from rasterio.vrt import WarpedVRT
from pathlib import Path
import os


# 1. Set up dask client. You will need to play around with the memory and number of workers based on your computer specs. 
## Also, the configuration will change for a cloud deployment. 
def start_dask():
    client = Client(
        n_workers=2,
        threads_per_worker=1,
        memory_limit="6GB",
    )
    print("Dashboard:", client.dashboard_link)
    return client


# 2. Load raster 
# The chunk size depends on a number of factors - see the slide presentation for more details. 
def load_raster(path):
    da_raster = (
        rioxarray.open_rasterio(path, chunks={"x": 1024, "y": 1024})
        .squeeze()
    )
    print("Raster dims:", da_raster.dims)
    print("Raster chunks:", da_raster.chunks)
    return da_raster


# 3. Reproject + Resample using WarpedVRT
def reproject_to_basegrid_vrt(
    src_raster_path,
    base_crs="EPSG:4326",
    base_res=(0.00083333333, -0.00083333333),
    resampling_method=Resampling.sum,
    out_chunks={"x": 1024, "y": 1024},
):
    """
    Create a WarpedVRT for the input raster and open it lazily with rioxarray.
    """
    with rasterio.open(src_raster_path) as src:
        if src.crs is None:
            raise ValueError("Source raster has no CRS.")

        left, bottom, right, top = src.bounds
        xres, yres = base_res
        dst_transform = from_origin(left, top, xres, abs(yres))

        dst_width = int(round((right - left) / xres))
        dst_height = int(round((top - bottom) / abs(yres)))

        vrt = WarpedVRT(
            src,
            crs=base_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=resampling_method,
            add_alpha=False,
        )

        da_vrt = rioxarray.open_rasterio(vrt, chunks=out_chunks).squeeze()

        x_coords = (np.arange(dst_width) + 0.5) * xres + left
        y_coords = top - (np.arange(dst_height) + 0.5) * abs(yres)

        da_vrt = da_vrt.assign_coords({"x": x_coords, "y": y_coords})
        da_vrt.rio.write_crs(base_crs, inplace=True)
        da_vrt.rio.write_transform(dst_transform, inplace=True)

        return da_vrt

# 4. Load and process raster based on folder type
def load_and_prepare_raster(
    raster_path,
    folder_type,
    base_crs="EPSG:4326",
    base_res=(0.00083333333, -0.00083333333),
):
    """
    Load and prepare raster based on folder type:
    - folder_1: Downsample with Resampling.sum
    - folder_2: Upsample with Resampling.nearest
    - folder_3: No reprojection, just replace nodata with 0
    """
    if folder_type == "folder_1":
        raster = reproject_to_basegrid_vrt(
            raster_path,
            base_crs=base_crs,
            base_res=base_res,
            resampling_method=Resampling.sum,
        )
    elif folder_type == "folder_2":
        raster = reproject_to_basegrid_vrt(
            raster_path,
            base_crs=base_crs,
            base_res=base_res,
            resampling_method=Resampling.nearest,
        )
    elif folder_type == "folder_3":
        raster = load_raster(raster_path)
        # Replace nodata values (-99999, -9999) with 0
        raster = raster.where((raster != -99999) & (raster != -9999), 0)
    else:
        raise ValueError(f"Unknown folder type: {folder_type}")
    
    return raster

# 5. Load admin3 zones from shapefile
def load_shapes(shp, raster):
    gdf = gpd.read_file(shp)

    if gdf.crs != raster.rio.crs:
        gdf = gdf.to_crs(raster.rio.crs)

    gdf["zone_id"] = range(len(gdf))
    shapes = [(geom, zid) for geom, zid in zip(gdf.geometry, gdf.zone_id)]
    return gdf, shapes


# 6. Rasterize Chunk
def rasterize_chunk(_, block_info=None, shapes=None, transform=None):
    info = block_info[0]
    (y0, y1), (x0, x1) = info["array-location"]

    height = y1 - y0
    width = x1 - x0

    block_transform = transform * transform.translation(x0, y0)

    xmin, ymin = block_transform * (0, height)
    xmax, ymax = block_transform * (width, 0)
    block_bounds = box(xmin, ymin, xmax, ymax)

    local_shapes = [(g, zid) for g, zid in shapes if g.intersects(block_bounds)]

    if not local_shapes:
        return np.full((height, width), -1, dtype="int32")

    return rasterize(
        local_shapes,
        out_shape=(height, width),
        transform=block_transform,
        fill=-1,
        dtype="int32",
    )


# 7. Build Zone Raster
def build_zone_raster(raster, shapes):
    raster_chunks = raster.data.chunks
      
    dummy = da.zeros(
        (raster.sizes["y"], raster.sizes["x"]),
        chunks=raster_chunks,
        dtype="int16"
    )

    zones = da.map_blocks(
        rasterize_chunk,
        dummy,
        dtype="int32",
        chunks=dummy.chunks,
        shapes=shapes,
        transform=raster.rio.transform(),
    )

    return xr.DataArray(
        zones, 
        dims=("y", "x"),
        coords={"y": raster.y, "x": raster.x}
    )

# 8. Block Stats
def block_stats(r_block, z_block):
    stats = {}
    if r_block.shape != z_block.shape:
        raise ValueError(f"Shape mismatch: raster {r_block.shape} vs zones {z_block.shape}")
    
    zones = np.unique(z_block)

    for z in zones:
        if z < 0:
            continue

        mask = z_block == z
        v = r_block[mask]

        if v.size == 0:
            continue

        valid_mask = np.isfinite(v)
        if v.dtype.kind == 'f':
            valid_mask &= (v != -99999) & (v != -9999) & (v != -3.4028235e+38)

        v_valid = v[valid_mask]
        if v_valid.size == 0:
            continue

        stats[int(z)] = {
            "sum": float(np.sum(v_valid, dtype=np.float64)),
            "count": int(v_valid.size),
            "min": float(np.min(v_valid)),
            "max": float(np.max(v_valid)),
        }

    return stats

# 9. Compute Zonal Stats
def compute_zonal_stats(raster, zones):
    from dask import delayed
    import dask
    
    if raster.data.chunks != zones.data.chunks:
        zones = zones.chunk(dict(zip(zones.dims, raster.data.chunks)))

    n_blocks_y = len(raster.data.chunks[0])
    n_blocks_x = len(raster.data.chunks[1])
    
    delayed_results = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            r_block = raster.data.blocks[i, j]
            z_block = zones.data.blocks[i, j]
            result = delayed(block_stats)(r_block, z_block)
            delayed_results.append(result)

    with ProgressBar():
        results = dask.compute(*delayed_results)

    merged = {}
    for block in results:
        if not block:
            continue
        for zid, st in block.items():
            if zid not in merged:
                merged[zid] = st
            else:
                merged[zid]["sum"] += st["sum"]
                merged[zid]["count"] += st["count"]
                merged[zid]["min"] = min(merged[zid]["min"], st["min"])
                merged[zid]["max"] = max(merged[zid]["max"], st["max"])

    for zid in merged:
        merged[zid]["mean"] = merged[zid]["sum"] / merged[zid]["count"] if merged[zid]["count"] > 0 else np.nan

    df = pd.DataFrame.from_dict(merged, orient="index")
    df.index.name = "zone_id"
    return df.reset_index()

# 10. Process Single Raster
def process_single_raster(
    raster_path,
    folder_type,
    shp_path,
    adm_gdf,
    base_crs="EPSG:4326",
    base_res=(0.00083333333, -0.00083333333),
):
    """Process a single raster file and return zonal statistics."""
    
    # Load and prepare raster based on folder type
    raster = load_and_prepare_raster(raster_path, folder_type, base_crs, base_res)
    
    # Load shapes (use cached adm_gdf if provided)
    if adm_gdf is None:
        adm, shapes = load_shapes(shp_path, raster)
    else:
        # Reuse loaded shapefile
        adm = adm_gdf.copy()
        if adm.crs != raster.rio.crs:
            adm = adm.to_crs(raster.rio.crs)
        shapes = [(geom, zid) for geom, zid in zip(adm.geometry, adm.zone_id)]
    
    # Build zone raster
    zones_da = build_zone_raster(raster, shapes)
    
    # Compute zonal stats
    df = compute_zonal_stats(raster, zones_da)
    
    # Add raster name column
    df["raster_file"] = Path(raster_path).stem
    
    # Merge with admin codes
    df = df.merge(adm[["zone_id", "ADM3_PCODE"]], on="zone_id", how="left")
    
    return df


# 11. Main Pipeline - Process All Folders
def run_pipeline():
    
    client = start_dask()
    
    # CONFIGURATION
    base_dir = r"path\to\base_dir\MOZ"
    
    folder_configs = {
        "folder_1": {
            "path": os.path.join(base_dir, "folder_1"),
            "pattern": "*.tif",
            "type": "folder_1"  # Downsample with sum
        },
        "folder_2": {
            "path": os.path.join(base_dir, "folder_2"),
            "pattern": "*.tif",
            "type": "folder_2"  # Upsample with nearest
        },
        "folder_3": {
            "path": os.path.join(base_dir, "folder_3"),
            "pattern": "*.tif",
            "type": "folder_3"  # No reprojection, encode nodata as 0
        },
    }
    
    shp_path = r"path\to\data\shapefiles\MOZ_ADM3.shp"
    output_dir = r"path\to\data\MOZ\zonalstats"
    
    base_crs = "EPSG:4326"
    base_res = (0.00083333333, -0.00083333333) # this was extracted from the base raster
    
    # Load shapefile once (reuse for all rasters)
    adm_gdf = gpd.read_file(shp_path)
    adm_gdf["zone_id"] = range(len(adm_gdf))
    
    # Process all folders sequentially
    all_results = []
    
    for folder_name, config in folder_configs.items():
        folder_path = config["path"]
        folder_type = config["type"]
        pattern = config["pattern"]
        
        # Get all raster files in folder
        raster_files = list(Path(folder_path).glob(pattern))
        
        if not raster_files:
            print(f" No raster files found in {folder_path}")
            continue
        
        # Process each raster in the folder
        for raster_file in raster_files:
            try:
                df = process_single_raster(
                    str(raster_file),
                    folder_type,
                    shp_path,
                    adm_gdf,
                    base_crs,
                    base_res,
                )
                df["folder"] = folder_name
                all_results.append(df)
            except Exception as e:
                print(f"ERROR processing {raster_file.name}: {e}")
                continue
    
    # Combine and save all results
    if all_results:
       
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "zonal_stats_all.csv")
        combined_df.to_csv(output_file, index=False)
      
        print(combined_df.head(10))
        
        # Also save separate files per folder
        for folder_name in combined_df["folder"].unique():
            folder_df = combined_df[combined_df["folder"] == folder_name]
            folder_output = os.path.join(output_dir, f"zonal_stats_{folder_name}.csv")
            folder_df.to_csv(folder_output, index=False)
            print(f"  ✓ Saved {folder_name}: {len(folder_df)} rows → {folder_output}")
    else:
        print("\n No results to save.")
    
    client.close()
    print("\n Pipeline completed!")

# 12. Entry point
if __name__ == "__main__":
    run_pipeline()