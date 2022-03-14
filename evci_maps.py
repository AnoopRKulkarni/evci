#!/usr/bin/env python
# coding: utf-8
#
# MPEN EVCI Tool SiteMap Viz Page
# Authors: Anoop R Kulkarni
# Version 4.0
# Mar 14, 2022

#@title Import libraries
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

import os
import pandas as pd

import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def app(site):
    #@title Define file paths
    INPUT_PATH = 'input/'+site['prefix']+'/'
    OUTPUT_PATH = 'output/'+site['prefix']+'/'

    if not os.path.exists(INPUT_PATH):
       os.makedirs(INPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
       os.makedirs(OUTPUT_PATH)
    
    #@title Read Master Data Excel Book
    newdata = pd.read_excel(INPUT_PATH + site['file'], sheet_name=None)
    datasheets = list(newdata.keys())
    
    # # State map
    st.sidebar.title("Selections")
    
    #@title Select map layers
    all_datasheets = datasheets
    sel_datasheets = st.sidebar.multiselect('Select Layers', all_datasheets, all_datasheets)
    
    #@title Read data from selected sheets only
    df = gpd.read_file(INPUT_PATH + site['gis']+'/'+site['gis']+'.shp')
    
    #@title Total Area
    area = df['geometry'].to_crs(6933).map(lambda p: p.area/ 1e6).loc[0]
    st.sidebar.header(f'Total Area: {area:.2f} sq-km')
    
    
    #@title Select Layers
    data = {}
    
    for s in all_datasheets:
      if s in sel_datasheets:
        data[s] = newdata[s]
        data[s]['Latitude'] = pd.to_numeric(data[s]['Latitude'])
        data[s]['Longitude'] = pd.to_numeric(data[s]['Longitude'])
        data[s]['geometry'] = [shapely.geometry.Point(xy) for xy in 
                              zip(data[s]['Longitude'], 
                                  data[s]['Latitude'])]
    
    #@title Create grid
    # total area for the grid
    xmin, ymin, xmax, ymax= df.total_bounds
    # how many cells across and down
    n_cells=30
    cell_size = (xmax-xmin)/n_cells
    # projection of the grid
    crs = df.crs
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append( shapely.geometry.box(x0, y0, x1, y1) )
    cell = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    #@title Create dataframes
    grid_df = gpd.overlay(df, cell, how='intersection')
    
    cell_area = area/grid_df.shape[0]
    if cell_area < 1:
       st.sidebar.subheader(f'Approx grid cell size: {cell_area*1e3:.2f} sq-m')
    else:
       st.sidebar.subheader(f'Approx grid cell size: {cell_area:.2f} sq-km')
    
    data_df = {}
    
    for s in data:
      data_df[s] = gpd.GeoDataFrame(data[s], 
                                    geometry=data[s]['geometry'])
    
    #@title Show map layers
    base = grid_df.plot(color='none', edgecolor='grey', alpha=0.4, figsize=(14,8))
    df.plot(ax=base, color='none', edgecolor='black')
    
    excel_sheets = list(set(sel_datasheets) & set(datasheets))
    for s in excel_sheets:
      data_df[s].plot(ax=base, markersize=50, legend=True, label=s) 
    
    plt.legend(bbox_to_anchor=(1,1), loc="upper left")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.tight_layout() 
    
    st.header("All sites")
    st.pyplot()
