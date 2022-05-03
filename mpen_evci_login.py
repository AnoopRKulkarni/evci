#
# MPEN EVCI Tool Landing Page
# Authors: Anoop R Kulkarni
# Version 4.0
# Mar 14, 2022

import evci_maps
import evci_analysis
import streamlit as st
import extra_streamlit_components as stx
import webbrowser
import requests
import toml

FLASK_APP = 'localhost:8080'
#FLASK_APP = 'tool.evci.in:8080'
STL_APP = 'http://localhost:8501'
#STL_APP = 'http://tool.evci.in:8501'

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

lounge = st.empty()
with lounge.container():
   st.image ('res/evci.jpg')

if 'loggedin' not in st.session_state:
   st.session_state.loggedin = False
if 'explored' not in st.session_state:
   st.session_state.explored = False
   
cookie_manager = stx.CookieManager()

if cookie_manager.get(cookie='mpen_evci_user') != None:
   st.session_state.loggedin = True

if st.session_state.loggedin:
   
   if st.sidebar.button('Log Out'):
      st.session_state.loggedin = False
      st.session_state.explored = False
      cookie_manager.delete(cookie='mpen_evci_user')
   else:
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

else:
   if st.button('Log In'):
      webbrowser.open(FLASK_APP)
