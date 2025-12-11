
"""
Large-Scale Geospatial Raster Processing Pipeline
==================================================
Process multiple raster datasets with heterogeneous characteristics:
- Downsample high-resolution data (folder_1)
- Upsample low-resolution data (folder_2)
- Clean already-aligned data (folder_3)

Compute zonal statistics for administrative boundaries using:
- Xarray: Labeled arrays with geospatial awareness
- Dask: Parallel, out-of-core computation
- WarpedVRT: Streaming reprojection without temp files
"""

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


# 1. Set up dask client. You will need to play around with the memory and number of workers based on your resources. 
## Also, the configuration will change for a cloud deployment. 
def start_dask():
    """
    Initialize Dask distributed client for parallel processing.
    
    Configuration choices:
    - n_workers=4: 
    - threads_per_worker=1: 
    - memory_limit="5GB": Hard limit prevents OOM, forces disk spilling
    
    Returns:
        Client: Dask distributed client instance
    """
    client = Client(
        n_workers=4,
        threads_per_worker=1,
        memory_limit="5GB",
    )
    print("Dashboard:", client.dashboard_link)
    return client


# 2. Load raster 
# The chunk size depends on a number of factors - see the slide presentation for more details. 
def load_raster(path):
    """
    Load raster file as chunked Xarray DataArray for lazy evaluation.
    
    Key features:
    - chunks={"x": 1024, "y": 1024}: Each chunk ~4MB for float32 data
    - squeeze(): Remove single-dimension entries (e.g., band dimension)
    - Lazy loading: Data read only when compute() is called
    
    Args:
        path: Path to raster file
        
    Returns:
        xr.DataArray: Chunked raster with geospatial metadata
    """
    da_raster = (
        rioxarray.open_rasterio(path, chunks={"x": 1024, "y": 1024})
        .squeeze()
    )
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
    Create WarpedVRT for on-demand, windowed reprojection without temp files.
    
    WarpedVRT enables:
    - Lazy reprojection: Transformation happens during chunk reads
    - No intermediate files: Zero disk I/O overhead
    - Memory efficient: Only processes requested windows
    - Dask compatible: Maintains lazy evaluation
    
    Process:
    1. Calculate destination grid parameters
    2. Create virtual reprojected raster (VRT)
    3. Open VRT with chunking (stays lazy)
    4. Assign proper coordinates and metadata
    
    Args:
        src_raster_path: Path to source raster
        base_crs: Target coordinate reference system
        base_res: Target resolution (x_res, y_res)
        resampling_method: How to resample (sum, nearest, bilinear, etc.)
        out_chunks: Chunk sizes for output array
        
    Returns:
        xr.DataArray: Virtually reprojected raster (lazy)
    """
    with rasterio.open(src_raster_path) as src:
        # validate source has CRS
        if src.crs is None:
            raise ValueError("Source raster has no CRS.")
        # get source bounds in CRS
        left, bottom, right, top = src.bounds
        # Calculate destination grid parameters
        xres, yres = base_res
        dst_transform = from_origin(left, top, xres, abs(yres))
        # calculate destination dimensions
        dst_width = int(round((right - left) / xres))
        dst_height = int(round((top - bottom) / abs(yres)))
        
        # create a WarpedVRT: virtual reprojected raster
        # This does not reproject data yet - it just creates a view
        vrt = WarpedVRT(
            src,
            crs=base_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
            resampling=resampling_method,
            add_alpha=False,
        )

        # open VRT with rioxarray, maintaining chunking
        # so the data stays lazy - reprojection happens during chunk reads
        da_vrt = rioxarray.open_rasterio(vrt, chunks=out_chunks).squeeze()

        # calculate coordinate arrays for pixel centers
        x_coords = (np.arange(dst_width) + 0.5) * xres + left
        y_coords = top - (np.arange(dst_height) + 0.5) * abs(yres)

        # assign coordinates to DataArray
        da_vrt = da_vrt.assign_coords({"x": x_coords, "y": y_coords})
        # write CRS and transform metadata
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
    Load and prepare raster based on source characteristics.
    
    Three processing modes:
    
    folder_1 (HIGH RESOLUTION → DOWNSAMPLE):
    - Use Resampling.sum to preserve totals (e.g., population counts)
    - Aggregates values when combining multiple pixels into one
    
    folder_2 (LOW RESOLUTION → UPSAMPLE):
    - Use Resampling.nearest to avoid interpolation artifacts
    - Maintains discrete values (e.g., land cover classes)
    
    folder_3 (ALREADY ALIGNED):
    - No reprojection needed
    - Just replace nodata values with 0
    
    Args:
        raster_path: Path to raster file
        folder_type: "folder_1", "folder_2", or "folder_3"
        base_crs: Target CRS for folders 1 and 2
        base_res: Target resolution for folders 1 and 2
        
    Returns:
        xr.DataArray: Prepared raster ready for analysis
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
    """
    Load shapefile and prepare for rasterization.
    
    Steps:
    1. Read shapefile with GeoPandas
    2. Reproject to match raster CRS if needed
    3. Assign unique zone IDs
    4. Create (geometry, zone_id) tuples for rasterization
    
    Args:
        shp: Path to shapefile
        raster: Reference raster for CRS matching
        
    Returns:
        tuple: (GeoDataFrame with zone_id, list of (geometry, zone_id) tuples)
    """
    gdf = gpd.read_file(shp)

    # reproject to match crs of the raster in case its different
    if gdf.crs != raster.rio.crs:
        gdf = gdf.to_crs(raster.rio.crs)

    # assign unique integer IDs to each zone
    gdf["zone_id"] = range(len(gdf))
    # create list of (geometry, value) tuples for rasterization
    shapes = [(geom, zid) for geom, zid in zip(gdf.geometry, gdf.zone_id)]
    return gdf, shapes


# 6. Rasterize Chunk
def rasterize_chunk(_, block_info=None, shapes=None, transform=None):
    """
    Rasterize vector geometries for a single chunk.
    
    Called by da.map_blocks for each chunk. Process:
    1. Extract chunk location from block_info
    2. Calculate chunk's geographic bounds
    3. Filter to geometries intersecting chunk
    4. Rasterize only relevant geometries
    
    This chunk-wise approach:
    - Processes only relevant geometries per chunk
    - Enables parallelization
    - Reduces memory usage
    
    Args:
        _: Dummy array (ignored, just for block structure)
        block_info: Dask block metadata (location, shape)
        shapes: List of (geometry, zone_id) tuples
        transform: Affine transform for full raster
        
    Returns:
        np.ndarray: Rasterized zone IDs for this chunk (-1 for background)
    """
    # extract chunk location in array coordinates
    info = block_info[0]
    (y0, y1), (x0, x1) = info["array-location"]

    height = y1 - y0
    width = x1 - x0

    # calculate chunk's geographic transform
    block_transform = transform * transform.translation(x0, y0)

    # calculate chunks's geographic bounds
    xmin, ymin = block_transform * (0, height)
    xmax, ymax = block_transform * (width, 0)
    block_bounds = box(xmin, ymin, xmax, ymax)

    # filter to geometries that intersect this chunk
    # this is key for performance: only process relevant geometries
    local_shapes = [(g, zid) for g, zid in shapes if g.intersects(block_bounds)]

    # if no geometries intersect, return background (-1)
    if not local_shapes:
        return np.full((height, width), -1, dtype="int32")

    # rasterize intersecting geometries
    return rasterize(
        local_shapes,
        out_shape=(height, width),
        transform=block_transform,
        fill=-1,
        dtype="int32",
    )


# 7. Build Zone Raster
def build_zone_raster(raster, shapes):
    """
    Create rasterized zone array with chunks EXACTLY matching raster.
    
    CRITICAL: Chunks must align between raster and zones for block-wise operations.
    Mismatched chunks cause shape errors during element-wise operations.
    
    Process:
    1. Extract raster's chunk structure
    2. Create dummy array with identical chunks
    3. Use map_blocks to rasterize each chunk
    4. Return as Xarray DataArray with matching coordinates
    
    Args:
        raster: Reference raster (provides chunk structure)
        shapes: List of (geometry, zone_id) tuples
        
    Returns:
        xr.DataArray: Rasterized zones with matching chunks
    """
    # get the raster's exact chunk structure
    # this is critical, zones must have identical chunks to raster
    raster_chunks = raster.data.chunks
      
    # create dummy array with matching chunk structure
    dummy = da.zeros(
        (raster.sizes["y"], raster.sizes["x"]),
        chunks=raster_chunks,
        dtype="int16"
    )

    # rasterize each chunk in parallel
    zones = da.map_blocks(
        rasterize_chunk,
        dummy,
        dtype="int32",
        chunks=dummy.chunks,
        shapes=shapes,
        transform=raster.rio.transform(),
    )

    # wrap in Xarray DataArray with matching coordinates
    return xr.DataArray(
        zones, 
        dims=("y", "x"),
        coords={"y": raster.y, "x": raster.x}
    )

# 8. Block Stats
def block_stats(r_block, z_block):
    """
    Compute zonal statistics for a single chunk.
    
    For each zone in the chunk:
    1. Extract raster values in that zone
    2. Filter nodata values (NaN, -9999, etc.)
    3. Compute sum, count, min, max
    
    Key considerations:
    - Use float64 for sum to prevent overflow (billions of pixels)
    - Filter multiple nodata representations
    - Handle empty zones gracefully
    
    Args:
        r_block: Raster values for this chunk
        z_block: Zone IDs for this chunk
        
    Returns:
        dict: {zone_id: {sum, count, min, max}} for zones in this chunk
    """
    stats = {}
    # validate block shapes match - the raster against the zones
    if r_block.shape != z_block.shape:
        raise ValueError(f"Shape mismatch: raster {r_block.shape} vs zones {z_block.shape}")
    
    # get unique zone IDs in this chunk
    zones = np.unique(z_block)

    for z in zones:
        if z < 0:
            continue

        # extract raster values for this zone
        mask = z_block == z
        v = r_block[mask]

        if v.size == 0:
            continue

        #filter no data values
        valid_mask = np.isfinite(v)
        if v.dtype.kind == 'f':
            valid_mask &= (v != -99999) & (v != -9999) & (v != -3.4028235e+38)

        v_valid = v[valid_mask]
        if v_valid.size == 0:
            continue

        # use float64 for sum to prevent overflow with large datasets
        stats[int(z)] = {
            "sum": float(np.sum(v_valid, dtype=np.float64)),
            "count": int(v_valid.size),
            "min": float(np.min(v_valid)),
            "max": float(np.max(v_valid)),
        }

    return stats

# 9. Compute Zonal Stats for all chunks
def compute_zonal_stats(raster, zones):
    """
    Compute zonal statistics using Dask delayed for parallel processing.
    
    Why delayed instead of map_blocks?
    - map_blocks expects array outputs
    - Our function returns dicts (non-standard)
    - delayed is more flexible for custom aggregations
    
    Process:
    1. Verify chunk alignment
    2. Create delayed task for each chunk pair
    3. Compute all tasks in parallel
    4. Merge results across chunks
    5. Calculate means from sum/count
    
    Args:
        raster: Raster data array
        zones: Zone ID array
        
    Returns:
        pd.DataFrame: Zonal statistics (zone_id, sum, count, min, max, mean)
    """
    from dask import delayed
    import dask
    
    # verify chunk alignment
    if raster.data.chunks != zones.data.chunks:
        zones = zones.chunk(dict(zip(zones.dims, raster.data.chunks)))

    # get number of chunks in each dimension
    n_blocks_y = len(raster.data.chunks[0])
    n_blocks_x = len(raster.data.chunks[1])
    
    # create delayed task for each chunk pair
    delayed_results = []
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            # lazy reference to chunks
            r_block = raster.data.blocks[i, j]
            z_block = zones.data.blocks[i, j]
            # wrap in delayed - creates task without executing
            result = delayed(block_stats)(r_block, z_block)
            delayed_results.append(result)

    # execute task in parallel
    with ProgressBar():
        results = dask.compute(*delayed_results)

    # merge results across all chunks
    merged = {}
    for block in results:
        if not block:
            continue
        for zid, st in block.items():
            if zid not in merged:
                merged[zid] = st
            
            else:
              # zone appears in multiple chunks - accumulate
                merged[zid]["sum"] += st["sum"]
                merged[zid]["count"] += st["count"]
                merged[zid]["min"] = min(merged[zid]["min"], st["min"])
                merged[zid]["max"] = max(merged[zid]["max"], st["max"])

    # calculate mean from sum and count
    for zid in merged:
        merged[zid]["mean"] = merged[zid]["sum"] / merged[zid]["count"] if merged[zid]["count"] > 0 else np.nan

    # convert to dataframe
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
    """
    Complete workflow for processing a single raster file.
    
    Steps:
    1. Load and prepare raster (reproject if needed)
    2. Load/reuse shapefile (with CRS matching)
    3. Build zone raster (rasterize polygons)
    4. Compute zonal statistics
    5. Add metadata (file name, admin codes)
    
    Args:
        raster_path: Path to raster file
        folder_type: "folder_1", "folder_2", or "folder_3"
        shp_path: Path to shapefile
        adm_gdf: Pre-loaded GeoDataFrame (for reuse) or None
        base_crs: Target CRS
        base_res: Target resolution
        
    Returns:
        pd.DataFrame: Zonal statistics with metadata
    """
    
    # Load and prepare raster based on folder type - this handles reprojection and resampling
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
    
    # Add source file name for tracking
    df["raster_file"] = Path(raster_path).stem
    
    # Merge with admin codes from shapefile
    df = df.merge(adm[["zone_id", "ADM3_PCODE"]], on="zone_id", how="left")
    
    return df


# 11. Main Pipeline - Process All Folders
def run_pipeline():
    
    """
    Main pipeline: Process all rasters from three folders sequentially.
    
    Pipeline structure:
    1. Start Dask client for parallel processing
    2. Configure folder locations and processing types
    3. Load shapefile ONCE (reuse for all rasters)
    4. Process each folder sequentially
    5. Within each folder, process rasters (parallelized via chunks)
    6. Combine results and save
    
    Error handling:
    - Isolate failures per file (one error doesn't stop pipeline)
    - Log errors but continue processing
    - Save partial results if some files succeed
    """
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
