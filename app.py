# app.py
import streamlit as st
from qa_app import answer

st.set_page_config(page_title="F‑16 Manual QA")
st.title("F‑16 Manual QA (GPU + Refine Chain)")

question = st.text_input("Ask a question about the F‑16 manual:")
if st.button("Submit") and question:
    with st.spinner("Thinking…"):
        response = answer(question)
    st.markdown(response)
