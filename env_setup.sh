#!/bin/bash

python -m venv env
source env/Scripts/activate
pip install dask geopandas rioxarray xarray pandas numpy rasterstats
