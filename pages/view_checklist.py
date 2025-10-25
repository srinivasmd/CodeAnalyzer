import streamlit as st
import json
import os

st.set_page_config(page_title="Security Checklist", layout="wide")
st.title("Security Checklist")

base_dir = os.path.dirname(os.path.dirname(__file__))  # get parent dir of pages/
checklist_path = os.path.join(base_dir, 'security-checklist.json')

if os.path.exists(checklist_path):
    with open(checklist_path, 'r', encoding='utf-8') as sf:
        checklist_content = json.load(sf)
        st.json(checklist_content)
else:
    st.warning("Security Checklist file not found. Please ensure 'security-checklist.json' exists in the project root.")