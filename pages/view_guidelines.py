import streamlit as st
import json
import os

st.set_page_config(page_title="PR Review Guidelines", layout="wide")
st.title("PR Review Guidelines")

base_dir = os.path.dirname(os.path.dirname(__file__))  # get parent dir of pages/
guidelines_path = os.path.join(base_dir, 'pr_review_guidelines.json')

if os.path.exists(guidelines_path):
    with open(guidelines_path, 'r', encoding='utf-8') as gf:
        guidelines_content = json.load(gf)
        st.json(guidelines_content)
else:
    st.warning("PR Review Guidelines file not found. Please ensure 'pr_review_guidelines.json' exists in the project root.")