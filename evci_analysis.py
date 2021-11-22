#!/usr/bin/env python
# coding: utf-8

# EVCI Analysis
# 
# 
# ## MP ENSYSTEMS

#@title Import libraries
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely

import os
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2, whiten
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

def app(site):
   global s_df_distances, s_df, s_u_df, final_list_of_sites
   global backoff_factor
   global initial_output_df, final_output_df

   #@title Define file paths
   INPUT_PATH = 'input/'+site['prefix']+'/'
   OUTPUT_PATH = 'output/'+site['prefix']+'/'
      
   #@title Read Master Data Excel Book
   newdata = pd.read_excel(INPUT_PATH + site['file'], sheet_name=None)
   datasheets = list(newdata.keys())

   #@title Define analytical model
   # Functions of the formulation to compute score, capex, opex and margin
   
   def score(j,i,hj,k,backoff=True, backoff_factor=1):
       distance_from_i = s_df_distances[s_df_distances > 0][i].sort_values()/1e3
       closer_to_i = distance_from_i[distance_from_i <= 5.0]
       try:
         congestion = float(s_df.loc[i]['Traffic congestion'])
       except:
         congestion = 1.0
       
       nw = qjworking[j][hj] * djworking[j][hj] * pj[k] * congestion
       nh = qjholiday[j][hj] * djholiday[j][hj] * pj[k] * congestion
       
       if backoff:
           for el in closer_to_i:
               nw *= (1 - np.exp(-el*backoff_factor))
               nh *= (1 - np.exp(-el*backoff_factor))
       
       tw = th = 0
       if (Cij[j][i] > 0): tw = nw * (tj[j]/Cij[j][i])
       if (Cij[j][i] > 0): th = nh * (tj[j]/Cij[j][i])
       
       uw = uh = tj[j]
       if (tw <= tj[j]): uw = tw 
       if (th <= tj[j]): uh = th
           
       vw = vh = 0
       if (tw > tj[j]): vw = (tw - tj[j]) * (Cij[j][i]/tj[j])
       if (th > tj[j]): vh = (th - tj[j]) * (Cij[j][i]/tj[j])
       
       norm_uw, norm_uh = uw/tj[j], uh/tj[j]
       norm_vw = vw/nw
       norm_vh = vh/nh
       
       return norm_uw, norm_uh, norm_vw, norm_vh
   
   def capex(i):
       retval = 0
       for j in C:
           retval += Cij[j][i]*Kj[j] + Wi[i] * di[i] * Cij[j][i]
       return retval
   
   def opex(i):
       op_e = 0
       op_l = 0
   
       for k in Gk:
           for j in C:
               for h in range(int(timeslots[j])):
                   sw, sh, _, _ = score (j,i,h,k)
                   op_e += 300 * Cij[j][i] * sw * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
                   op_e +=  65 * Cij[j][i] * sh * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
       op_l = Li[i] * Ai[i] + CH[i] + CK[i]
       return op_e + op_l
       
   def margin(i):
       margin_e = 0
       margin_l = 0
   
       for k in Gk:
         for j in C:
             for h in range(int(timeslots[j])):
                 sw, sh, _, _ = score (j,i,h,k)
                 margin_e += 300 * Cij[j][i] * sw * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
                 margin_e +=  65 * Cij[j][i] * sh * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
       margin_l = Bi[i] * Ai[i] + MH[i] + MK[i]
       return margin_e + margin_l
   
   def run_analysis(backoff_factor=1):
     global s_df_distances
     u_df = pd.DataFrame(columns=['utilization', 
                                'unserviced', 
                                'capex', 
                                'opex', 
                                'margin', 
                                'max vehicles', 
                                'estimated vehicles'
                                ])    
     s_df_crs = gpd.GeoDataFrame(s_df, crs='EPSG:4326')
     s_df_crs = s_df_crs.to_crs('EPSG:5234')
     s_df_distances = s_df_crs.geometry.apply(lambda g: s_df_crs.distance(g))      
   
     st.write("Running analysis on selected sites..")
     progressbar = st.progress(0)
   
     for i in range(Nc):
       max_vehicles = 0
       for j in C:
         max_vehicles += timeslots[j]*Cij[j][i]
       op_e = 0
       op_l = 0
       margin_e = 0
       margin_l = 0
       year_u_avg = np.array([])
       year_v_avg = np.array([])
       for k in Gk:
         chargertype_u_avg = np.array([])
         chargertype_v_avg = np.array([])
         for j in C:
           uw_day_avg = np.array([])
           uh_day_avg = np.array([])
           vw_day_avg = np.array([])
           vh_day_avg = np.array([])              
           for h in range(int(timeslots[j])):
             uw, uh, vw, vh = score (j,i,h,k, backoff_factor=backoff_factor)
             uw_day_avg = np.append(uw_day_avg, uw)
             uh_day_avg = np.append(uh_day_avg, uh)
             vw_day_avg = np.append(vw_day_avg, vw)
             vh_day_avg = np.append(vh_day_avg, vh)                  
             op_e += 300 * Cij[j][i] * uw * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])
             op_e +=  65 * Cij[j][i] * uh * tj[j] * Dj[j] * (l[j][h] * Eg[j][h] + (1-l[j][h]) * Er[j][h])            
             margin_e += 300 * Cij[j][i] * uw * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])
             margin_e +=  65 * Cij[j][i] * uh * tj[j] * Dj[j] * (l[j][h] * Mg[j][h] + (1-l[j][h]) * Mr[j][h])        
           weighted_u = (300.0*uw_day_avg.mean() + 65.0*uh_day_avg.mean()) / 365.0
           weighted_v = (300.0*vw_day_avg.mean() + 65.0*vh_day_avg.mean()) / 365.0
           chargertype_u_avg = np.append(chargertype_u_avg, weighted_u)
           chargertype_v_avg = np.append(chargertype_v_avg, weighted_v)
         year_u_avg = np.append(year_u_avg, chargertype_u_avg.mean())
         year_v_avg = np.append(year_v_avg, chargertype_v_avg.mean())
         op_l += Li[i] * Ai[i] + CH[i] + CK[i]
         margin_l += Bi[i] * Ai[i] + MH[i] + MK[i]
       site_capex = capex(i)
       estimated_vehicles = np.round(year_u_avg.mean()*max_vehicles,0)
       u_df.loc[i] = [ year_u_avg.mean(), 
                      year_v_avg.mean(),
                      site_capex,
                      op_e + op_l,
                      margin_e + margin_l,
                      max_vehicles,
                      estimated_vehicles
                      ]
       progressbar.progress((i+1)/Nc)
     return u_df
         
   st.sidebar.title("Selections")
   
   #@title Select sheets for analysis
   all_datasheets = datasheets
   sel_datasheets = st.sidebar.multiselect('Select Layers', all_datasheets, all_datasheets)
   
   #@title Read required data sheets only
   df = gpd.read_file(INPUT_PATH + site['gis']+'/'+site['gis']+'.shp')
   
   data = {}
   
   for s in sel_datasheets:
     data[s] = newdata[s]
     data[s]['Name'] = data[s]['Name']
     data[s]['Sheet'] = s
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
   
   grid_df = gpd.overlay(df, cell, how='intersection')
   
   data_df = {}
   
   for s in data:
     data_df[s] = gpd.GeoDataFrame(data[s], 
                                   geometry=data[s]['geometry'])
   
   #@title Prepare data
   s_df = pd.DataFrame(columns=['Name', 'Sheet',
                                'Latitude', 'Longitude',
                                'Traffic congestion',
                                'year 1',
                                'kiosk hoarding',
                                'hoarding margin',
                                'geometry'])
   
   for s in sel_datasheets:
     s_df = s_df.reset_index(drop=True)
     for i in range(data_df[s].shape[0]):
         s_df.loc[i+s_df.shape[0]] = [
               data_df[s].loc[i].Name, 
               s,
               data_df[s].loc[i].Latitude, 
               data_df[s].loc[i].Longitude, 
               data_df[s].loc[i]['Traffic congestion'],
               data_df[s].loc[i]['Year for Site recommendation'],
               data_df[s].loc[i]['Hoarding/Kiosk (1 is yes & 0 is no)'],
               data_df[s].loc[i]['Hoarding Margin (30% for year 1 and 15% for year 2 of base cost 9 lakhs)'],
               data_df[s].loc[i].geometry
           ]
   
   s_df = s_df[s_df['year 1'] == 1]
   s_df = s_df.reset_index(drop=True)
   Nc = s_df.shape[0]
   
   st.sidebar.header(f'Number of sites: {Nc}')
   
   
   #@title Choose charger types for installation
   # To be moved to a config file eventually
   
   M = {'2W', '4WS', '4WF'}
   
   choose_chargers = st.sidebar.multiselect('Select Chargers', M, M)
   
   years_of_analysis = st.sidebar.slider('Years for analysis', 1, 3, 3, 1)
   
   #@title Global Parameters
   C = set(choose_chargers)
   
   capex_2W = st.sidebar.slider('2W Capex (Rs)', 10000, 100000, 25000, 5000)
   capex_4W_slow_charging = st.sidebar.slider('4W Capex Slow Charging (Rs)', 50000, 200000, 100000, 5000)
   capex_4W_fast_charging = st.sidebar.slider('4W Capex Fast Charging (Rs)', 700000, 1200000, 900000, 50000)
   
   Kj = {'2W': capex_2W, '4WS': capex_4W_slow_charging, '4WF': capex_4W_fast_charging}
   Dj = {'2W': 2.2, '4WS': 22, '4WF': 60}
   Hj = {'2W': 5, '4WS': 15, '4WF': 15}
   Qj = {'2W': 1250, '4WS': 9143, '4WF': 18286}
   tj = {'2W': 1, '4WS': 1.5, '4WF': 0.5}
   Mj = {'2W': 500, '4WS': 500, '4WF': 500}
   
   Gk = {1: 1.0, 2: 1.0, 3: 1.0}
   K = years_of_analysis
   
   N = 500
   Ng = 0
   timeslots = {k: 24/v for k, v in tj.items()}
   
   Gi = [0]*Nc
   di = [0]*Nc
   Wi = [0]*Nc
   Ri = [0]*Nc
   Ai = [50]*Nc
   Li = [1500]*Nc
   Bi = [0.25 * 3.5 * 24 * 365]*Nc # 25% of Rs 3.5/KWh per year
   
   Eg = {k: [5.5] * int(v) for k, v in timeslots.items()}
   Er = {k: [0] * int(v) for k, v in timeslots.items()}
   Mg = {k: [5.5 * 0.15] * int(v) for k, v in timeslots.items()}
   Mr = {k: [0] * int(v) for k, v in timeslots.items()}
   l  = {k: [1] * int(v) for k, v in timeslots.items()}
   
   hoarding_cost = st.sidebar.slider('Hoarding cost (Rs)', 500000, 1000000, 900000, 50000)
   kiosk_cost = st.sidebar.slider('Kiosk cost (Rs)', 100000, 200000, 180000, 20000)
   
   CH = [hoarding_cost]*Nc
   CK = [kiosk_cost]*Nc
   
   MH = [s_df.loc[i]['hoarding margin'] for i in range(Nc)]
   MK = [0.15]*Nc
   
   #Traffic Model
   
   year1_conversion = 0.01 #@param {type:"slider", min:0, max:1, step:0.01}
   year2_conversion = 0.03 #@param {type:"slider", min:0, max:1, step:0.01}
   year3_conversion = 0.05 #@param {type:"slider", min:0, max:1, step:0.01}
   
   pj = {1: year1_conversion, 
         2: year2_conversion, 
         3: year3_conversion}
   
   Pj = max(pj.values()) 
   
   # peak vehicles through crowded junctions in a day ~ 1.5L
   
   peak_traffic = {}
   peak_traffic['pmc'] = [
            4826, 4826, 5228, 5228, 5228, 5630, 6434, 6836, 6836, 
            6434, 6032, 6032, 6032, 6032, 6434, 6836, 7239, 8043, 
            8043, 8043, 6836, 6032, 5630, 5228       
   ]
   peak_traffic['pcmc'] = [
            4826, 4826, 5228, 5228, 5228, 5630, 6434, 6836, 6836, 
            6434, 6032, 6032, 6032, 6032, 6434, 6836, 7239, 8043, 
            8043, 8043, 6836, 6032, 5630, 5228       
   ]
   peak_traffic['goa'] = [
            450, 450, 220, 240, 250, 1000, 1500, 5750, 5750, 
            5750, 2290, 1500, 1500, 1500, 2200, 2200, 5750, 
            5750, 5750, 2200, 1000, 1000, 500, 500
   ]

   peak_traffic = peak_traffic[site['prefix']]
                 
   # Average traffic approx 80% of peak
   avg_traffic = [i*.8 for i in peak_traffic]
   # 2W and 4W assumed to be 60% and 20% respectively
   avg_traffic_2W = [i*.6 for i in avg_traffic]
   avg_traffic_4W = [i*.2 for i in avg_traffic]
   djworking_hourly_2W = [i/5 for i in avg_traffic_2W]
   djworking_hourly = [i/5 for i in avg_traffic_4W]
   djworking_half_hourly = [val for val in djworking_hourly 
                            for _ in (0, 1)]
   djworking_one_and_half_hourly = list(np.mean(np.array(djworking_half_hourly).reshape(-1, 3), axis=1))
   
   djworking = {}
   djworking['2W'] = [np.round(i,0) for i in djworking_hourly_2W]
   djworking['4WF'] = [np.round(i,0) for i in djworking_half_hourly]
   djworking['4WS'] = [np.round(i,0) for i in djworking_one_and_half_hourly]
   
   holiday_percentage = st.sidebar.slider('Holiday traffic percentage', 0, 100, 30, 10)/100
   
   djholiday = {}
   djholiday['2W'] = [np.round(i*holiday_percentage,0) for i in djworking_hourly_2W]
   djholiday['4WF'] = [np.round(i*holiday_percentage,0) for i in djworking_half_hourly]
   djholiday['4WS'] = [np.round(i*holiday_percentage,0) for i in djworking_one_and_half_hourly]
   
   fast_charging = st.sidebar.slider('Fast Charging Conversion %', 0, 100, 8, 1)/100
   slow_charging = st.sidebar.slider('Slow Charging Conversion %', 0, 100, 2, 1)/100
   
   qjworking = {'4WS': [slow_charging] * int(timeslots['4WS']), 
                '4WF': [fast_charging] * int(timeslots['4WF']), 
                '2W' : [fast_charging + slow_charging] * int(timeslots['2W']), }
   qjholiday = {'4WS': [slow_charging] * int(timeslots['4WS']), 
                '4WF': [fast_charging] * int(timeslots['4WF']), 
                '2W' : [fast_charging + slow_charging] * int(timeslots['2W']), }
   
   Cij = {'2W': [4]*Nc, '4WS': [1]*Nc, '4WF':[1]*Nc}
   
   
   # In[ ]:
   
   
   #@title Typical vehicle traffic profile (assumed)
   st.header('Typical Traffic profile (Assumed)')
   plt.plot(djworking_hourly, label='4W')
   plt.plot(djworking_hourly_2W, label='2W')
   plt.plot(peak_traffic,label='Peak Traffic Volume')
   plt.xlabel('Time of day (24 hour format)')
   plt.ylabel('Number of vehicles')
   #plt.legend(bbox_to_anchor=(1,1), loc="upper left")
   plt.legend()
   
   st.pyplot()
   
   #@title Compute scores
   
   backoff_factor = st.sidebar.slider('Backoff factor', 1,5,2,1)
   clustering_cutoff = st.sidebar.slider('Clustering cutoff', 0.,0.1,0.01,0.01)
   
   with st.form("initial_analysis_form"):
      st.header("Initial Analysis")
      st.markdown("Please select the parameters from the left selection panel and then click the button **'Run Analysis'**")
      submitted = st.form_submit_button("Run Analysis")
      if submitted:
         u_df = run_analysis(backoff_factor=backoff_factor)      
         
         #@title Plot utilization histogram
         st.subheader("Utilization Histograms")
         fig, ax = plt.subplots(1,2, figsize=(8,3))
         ax[0].set_xlabel('Utilization')
         ax[1].set_xlabel('Unserviced')
         ax[0].set_ylabel('Count')
         ax[1].set_ylabel('Count')
         u_df.utilization.hist(ax=ax[0])
         u_df.unserviced.hist(ax=ax[1])
         st.pyplot()
         
         #@title Compute metrics
         capex, opex, margin = st.columns(3)
         
         with capex:
            st.header("Capex")
            st.subheader(f'{sum(u_df.capex)/1e7:.2f} Cr Rs')
   
         with opex:
            st.header("Opex")
            st.subheader(f'{sum(u_df.opex)/1e7:.2f} Cr Rs')
   
         with margin:
            st.header("Margin")
            st.subheader(f'{sum(u_df.margin)/1e7:.2f} Cr Rs')
         
         #@title Prepare data
         s_u_df = s_df.copy()
         
         s_u_df['utilization'] = u_df.utilization
         s_u_df['unserviced'] = u_df.unserviced
         s_u_df['capex'] = u_df.capex
         s_u_df['opex'] = u_df.opex
         s_u_df['margin'] = u_df.margin
         s_u_df['max vehicles'] = u_df['max vehicles']
         s_u_df['estimated vehicles'] = u_df['estimated vehicles']
         
         
         #@title Save initial analysis to Excel
         output_df = s_u_df.copy()
         output_df.drop('geometry', axis=1, inplace=True)
         output_df.to_excel(OUTPUT_PATH + 'initial_output.xlsx')
         initial_output_df = output_df.copy()
         #st.write(output_df.head())
         
         #@title Sites marked with utilization
         st.subheader("Site Utilization")
         base = grid_df.plot(color='none', edgecolor='grey', alpha=0.4) 
         df.plot(ax=base, color='none', edgecolor='black')
         
         tmp_df = gpd.GeoDataFrame(s_u_df)
         tmp_df.plot(ax=base, column='utilization', cmap='jet', markersize=50, legend=True) 
             
         plt.tight_layout()
         st.pyplot()

   # ## Threholding and Clustering
   # 
   # Using hierarchical clustering
         
   with st.form("clustering_form"):
      st.header("Clustering")
      st.markdown("Please click the button **'Find Clusters'** if you want to see effect of clustering")
      submitted = st.form_submit_button("Find Clusters")
      if submitted:
         #@title Threshold and cluster
         threshold_and_cluster = True
         clustering_candidates = s_u_df[(s_u_df.utilization <= 0.2) & (s_u_df['year 1'] == 1)]
         points = np.array((clustering_candidates.apply(lambda x: list([x['Latitude'], x['Longitude']]),axis=1)).tolist())
         Z = linkage (points, method='complete', metric='euclidean');
         plt.figure(figsize=(14,8))
         dendrogram(Z);
         st.pyplot()

         clusters = fcluster(Z, t=clustering_cutoff, criterion='distance')
         clustered_candidates = gpd.GeoDataFrame(clustering_candidates)
         base = grid_df.plot(color='none', alpha=0.2, edgecolor='black', figsize=(8,8))
         clustered_candidates.plot(ax=base, column=clusters)
         st.pyplot()
         
         # ### Final site selections
         
         #@title Build final list of sites
         confirmed_sites = s_u_df[s_u_df.utilization > 0.2]
         if threshold_and_cluster:
             val, ind = np.unique (clusters, return_index=True)
             clustered_sites = clustered_candidates.reset_index(drop=True)
             clustered_sites = clustered_sites.iloc[clustered_sites.index.isin(ind)]
             final_list_of_sites = pd.concat([confirmed_sites, clustered_sites], axis=0)
         else:
             final_list_of_sites = confirmed_sites.copy()
         
         st.subheader(f'Total sites shortlisted: {final_list_of_sites.shape[0]}')
         
   # ## Phase II - Final Analysis
         
   with st.form("final_analysis_form"):
      st.header("Final Analysis")
      st.markdown("Please select the parameters from the left selection panel and then click the button **'Run Final Analysis'**")
      submitted = st.form_submit_button("Run Final Analysis")
      if submitted:
         s_df = final_list_of_sites.copy()
         s_df = s_df.reset_index(drop=True)
         Nc = s_df.shape[0]
         st.subheader(f'Total shortlisted sites: {Nc}')
         
         #@title Compute scores
         u_df = pd.DataFrame(columns=['utilization', 'unserviced'])    
         s_df_crs = gpd.GeoDataFrame(s_df, crs='EPSG:4326')
         s_df_crs = s_df_crs.to_crs('EPSG:5234')
         s_df_distances = s_df_crs.geometry.apply(lambda g: s_df_crs.distance(g))    
         
         progressbar = st.progress(0)
         u_df = run_analysis(backoff_factor=backoff_factor)      
         
         #@title Plot utilization histogram
         fig, ax = plt.subplots(1,2, figsize=(8,3))
         ax[0].set_xlabel('Utilization')
         ax[1].set_xlabel('Unserviced')
         ax[0].set_ylabel('Count')
         ax[1].set_ylabel('Count')
         u_df.utilization.hist(ax=ax[0])
         u_df.unserviced.hist(ax=ax[1])
         st.pyplot()
         
         #@title Compute metrics
         capex, opex, margin = st.columns(3)
         
         with capex:
            st.header("Capex")
            st.subheader(f'{sum(u_df.capex)/1e7:.2f} Cr Rs')
  
         with opex:
            st.header("Opex")
            st.subheader(f'{sum(u_df.opex)/1e7:.2f} Cr Rs')
 
         with margin:
            st.header("Margin")
            st.subheader(f'{sum(u_df.margin)/1e7:.2f} Cr Rs')
         
         #@title Prepare output
         output_df = s_df.copy()
         output_df.drop('geometry', axis=1, inplace=True)
         output_df.head()
         
         #@title Save final analysis to Excel
         output_df.to_excel(OUTPUT_PATH + 'final_output.xlsx')
         final_output_df = output_df.copy()

   if 'initial_output_df' in globals():
      file_path = OUTPUT_PATH + 'initial_output.xlsx'
      with open(file_path, 'rb') as f:
          st.download_button(label = 'Download Initial Output', data = f, file_name = 'initial_output.xlsx', 
                             mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')    

   if 'final_output_df' in globals():
      file_path = OUTPUT_PATH + 'final_output.xlsx'
      with open(file_path, 'rb') as f:
          st.download_button(label = 'Download Final Output', data = f, file_name = 'final_output.xlsx', 
                             mime = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')    
