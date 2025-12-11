# PyDataGlobal 2025 Talk - Engineering large-scale geospatial raster with Xarray and Dask
This repo contains materials for my PyData Global 2025 talk on building scalable geospatial raster processing pipelines using *Xarray*, *Dask*.  
The talk focuses on *engineering principles, parallel processing, and memory-safe raster operations* especially when working with very large rasters.  

![](www/pipeline)


# Repository structure

```
.
├── huge_rasters_using_xarraydask.py      # Main processing pipeline
├── env_setup.sh                          # Environment setup script
├── PyDataGlobal.qmd                      # Quarto presentation
└── www/                                  # Slide images and diagrams

```

# **What the `huge_rasters_using_xarraydask.py` Script Does**

This script demonstrates a **full, production-style geospatial raster processing workflow** using chunked computation and distributed processing.

It covers the full pipeline:

## **1. Start a Dask Distributed Client**

* Creates a local Dask cluster
* Lets you tune workers/threads/memory for your machine

## **2. Load Huge Rasters Using Chunked Xarray + Dask**

* Rasters are opened lazily using `rioxarray.open_rasterio()`
* Uses configurable chunk sizes (e.g., 1024×1024)
* Avoids loading full rasters into memory
* Immediately gives visibility on:

  * Raster dimensions
  * Internal Dask chunking

## **3. Efficient Reprojection and Resampling with WarpedVRT**

Large rasters often exceed available RAM.
Here we use **Rasterio’s WarpedVRT**:

* Reprojects on-the-fly
* Lazily streams blocks instead of loading everything
* Supports different resampling for different folders:

  * **SUM** (downsampling for intensity-like rasters)
  * **NEAREST** (upsampling for categorical rasters)

The script reprojects into a **base grid** defined by:

* CRS: `EPSG:4326`
* Resolution: `(0.00083333333, -0.00083333333)`
  (≈ ~ 100m)


## **4. Preprocessing Rules Per Folder**

Each folder in the project uses a different workflow:

| Folder     | Operation    | Description                                 |
| ---------- | ------------ | ------------------------------------------- |
| `folder_1` | Downsample   | Weighted aggregation using `Resampling.sum` |
| `folder_2` | Upsample     | Nearest-neighbour upsampling                |
| `folder_3` | Clean nodata | No reprojection — nodata replaced with 0    |

The pipeline automatically chooses the correct approach.

## **5. Load and Prepare Admin Boundaries**

* Loads an ADM3 shapefile once
* Ensures CRS matches raster CRS
* Assigns each polygon a `zone_id`
* Prepares shape tuples for rasterization

## **6. Build a Zone Raster (Chunk-Aware Rasterization)**

This is one of the hardest parts of large-scale zonal statistics.

The script performs **chunk-by-chunk rasterization**:

* Each Dask block determines its own spatial extent
* Only polygons intersecting the block are rasterized
* Produces a zone raster perfectly aligned with the reprojected/processed raster
* Avoids creating a massive in-memory label raster

This is essential for rasters with **tens of thousands of chunks**.

## **7. Compute Zonal Statistics Block-by-Block**

A custom zonal statistics engine runs with:

* `dask.delayed`
* One task per (raster_chunk × zone_chunk)
* Automatic merging of statistics across blocks
* Handles nodata values correctly
* Computes:

  * Sum
  * Count
  * Min
  * Max
  * Mean

This avoids tools that require full in-memory rasters (e.g., rasterstats, geopandas overlay).


## **8. Process All Rasters in All Folders**

The pipeline:

1. Iterates through folders
2. Loads all `.tif` and `.tiff` rasters
3. Applies folder-specific preprocessing
4. Rasterizes zones
5. Computes zonal statistics
6. Merges results with ADM3 attributes
7. Saves both:

   * A combined CSV for all folders
   * A per-folder CSV

## **9. Output Files**

Outputs are written to:

```
.../zonalstats/zonal_stats_all.csv
.../zonalstats/zonal_stats_folder_1.csv
.../zonalstats/zonal_stats_folder_2.csv
.../zonalstats/zonal_stats_folder_3.csv
```

---

# **Why This Workflow Matters for Huge Rasters**

This approach is built for:

* **10–50GB+ rasters**
* **High-resolution climate/agriculture datasets**
* **SDG-level national statistics**
* **Raster stacks with hundreds of layers**

It uses:

* Lazy loading
* Chunkwise reprojection
* Chunkwise rasterization
* Distributed compute graphs
* Memory-safe processing

Everything is built around avoiding:

 Full raster reads
 Single-machine memory bottlenecks
 Inconsistent grids
 CRS mismatch issues

# To run the pipeline 
1. Install dependencies
```{bash}
bash env_setup.sh 
```


2. Edit paths in run_pipeline()

3. Download geospatial data from HDX(Population density maps), WorldPop harmonised covariates(https://hub.worldpop.org/project/categories?id=14), and Malaria atlas (https://data.malariaatlas.org/maps?layers=Accessibility:202001_Global_Motorized_Travel_Time_to_Healthcare,Malaria:202206_Global_Pf_Parasite_Rate&extent=-11815912.856289707,-6356003.33856192,28286163.259866484,14615055.359158086)
4. Run the *huge_rasters_using_xarraydask.py*