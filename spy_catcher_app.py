# spy_catcher_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Spy Catcher", layout="wide")

st.title("🔍 Spy Catcher: Markov Chain Simulation")
st.markdown("""
Use this app to simulate a spy hiding in South America using a Markov Chain. 
Each day, the spy moves randomly to one of the neighboring countries. This app shows the long-term probability of the spy being in each country.
""")

# --- Define Countries and Neighbors ---
neighbors = {
    'Argentina': ['Chile', 'Bolivia', 'Paraguay', 'Brazil', 'Uruguay'],
    'Bolivia': ['Peru', 'Brazil', 'Paraguay', 'Argentina', 'Chile'],
    'Brazil': ['Uruguay', 'Argentina', 'Paraguay', 'Bolivia', 'Peru', 'Colombia', 'Venezuela', 'Guyana', 'Suriname', 'French Guiana'],
    'Chile': ['Peru', 'Bolivia', 'Argentina'],
    'Colombia': ['Venezuela', 'Brazil', 'Peru', 'Ecuador'],
    'Ecuador': ['Colombia', 'Peru'],
    'French Guiana': ['Suriname', 'Brazil'],
    'Guyana': ['Venezuela', 'Brazil', 'Suriname'],
    'Paraguay': ['Bolivia', 'Brazil', 'Argentina'],
    'Peru': ['Ecuador', 'Colombia', 'Brazil', 'Bolivia', 'Chile'],
    'Suriname': ['Guyana', 'Brazil', 'French Guiana'],
    'Uruguay': ['Argentina', 'Brazil'],
    'Venezuela': ['Colombia', 'Brazil', 'Guyana']
}

countries = list(neighbors.keys())
n = len(countries)

# --- Build Transition Matrix ---
P = np.zeros((n, n))
for i, c in enumerate(countries):
    for neighbor in neighbors[c]:
        j = countries.index(neighbor)
        P[i, j] = 1 / len(neighbors[c])

# --- Calculate Stationary Distribution ---
vals, vecs = np.linalg.eig(P.T)
vec = vecs[:, np.isclose(vals, 1)]
stationary = vec[:, 0].real
stationary /= stationary.sum()

# --- Display Full Transition Matrix ---
st.subheader("📋 Full Transition Matrix")
st.dataframe(pd.DataFrame(P, index=countries, columns=countries).round(3))

# --- Plot Stationary Distribution ---
st.subheader("📊 Long-term Probability Distribution")
df_pi = pd.DataFrame({"Country": countries, "Probability": stationary})
df_pi_sorted = df_pi.sort_values("Probability", ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=df_pi_sorted, x="Probability", y="Country", palette="viridis", ax=ax)
ax.set_title("Probability of Spy Being in Each Country (Long-Term)")
ax.set_xlim(0, df_pi_sorted["Probability"].max() * 1.1)
st.pyplot(fig)

# --- Display Table ---
st.subheader("🧮 Stationary Distribution Table")
st.dataframe(df_pi_sorted.set_index("Country").style.format("{:.3f}"))

# --- Add Result Analysis ---
st.subheader("🕵️ Result Analysis")
most_likely_country = df_pi_sorted.iloc[0]["Country"]
most_likely_prob = df_pi_sorted.iloc[0]["Probability"]
neighbors_count = len(neighbors[most_likely_country])
st.markdown(f"""
**Most Likely Country:** **{most_likely_country}**  
**Probability:** **{most_likely_prob:.3f}**  

👉 The spy is most likely to be in **{most_likely_country}**, which has the highest long-term probability. This is likely because it is connected to **{neighbors_count}** other countries, allowing frequent transitions in and out. Countries with more neighbors tend to accumulate higher long-term probabilities in a uniform Markov Chain model.
""")
