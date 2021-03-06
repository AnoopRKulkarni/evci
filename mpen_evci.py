#
# MPEN EVCI Tool Landing Page
# Authors: Anoop R Kulkarni
# Version 4.0
# Mar 14, 2022

import evci_maps
import evci_analysis
import streamlit as st
import toml

sites = {}
sites['PCMC'] = {'prefix':'pcmc', 'file': 'PCMC Mastersheet_3Nov21.xlsx', 'gis':'PCMC_Wards'}
sites['Pune'] = {'prefix':'pmc', 'file': 'Pune site mastersheet_3Nov21.xlsx', 'gis':'PMC'}
sites['Goa']  = {'prefix':'goa', 'file': 'Master sheet_EVCI Locations_29Sep21_Final.xlsx', 'gis':'Goa'}

PAGES = {
    "Site Visualization": evci_maps,
    "Utilization Analysis": evci_analysis,
}

header = st.container()

primaryColor = toml.load("config.toml")['theme']['primaryColor']
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; border-radius:20px 20px 20px 20px; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

with header:
   mpen, title, ae = st.columns([1,5,1])
   with mpen:
      st.image('res/logo-mpen.png', width=200)
   with ae:
      st.image('res/logo-ae.png', width=100)

st.title("EVCI Site Planner")

if 'explored' not in st.session_state:
   st.session_state.explored = False

lounge = st.empty()
with lounge.container():
      st.image ('res/evci.jpg')

      st.sidebar.title('Site Selection')
      selection = st.sidebar.radio("Select Site", sites.keys())
   
      explored = st.sidebar.button("Explore!")
   
      if explored:
         st.session_state.explored = True
      
      if not st.session_state.explored:
         with lounge.container():
            st.image('res/evci.jpg')
   
      if st.session_state.explored:
         sel_category = st.sidebar.radio("Select Category", list(PAGES.keys()))
         lounge.empty()
         with lounge.container():
            PAGES[sel_category].app(sites[selection])
