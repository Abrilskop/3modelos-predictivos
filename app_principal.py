import streamlit as st

pages = [
    st.Page(
        "regresion-lineal-california-main/app.py",
        title="Caso 1: California Housing",
        icon="🏠",
        default=True,
    ),
    st.Page(
        "caso2_wine/app_wine.py",
        title="Caso 2: Wine Quality",
        icon="🍷",
    ),
    st.Page(
        "caso3_diabetes/app_diabetes.py",
        title="Caso 3: Diabetes",
        icon="🩺",
    ),
]

pg = st.navigation(pages)
pg.run()
