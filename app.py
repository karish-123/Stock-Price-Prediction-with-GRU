import streamlit as st
import plotly.graph_objects as go
from gru import get_predictions, train_and_save_model
import os
import random

# Load custom CSS
with open("neon.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Inject twinkling stars into the background
stars_html = '<div id="stars">'
for _ in range(50):
    top = random.randint(1, 99)
    left = random.randint(1, 99)
    size = random.uniform(1.5, 2.5)
    stars_html += f'<div class="star" style="top:{top}%; left:{left}%; width:{size}px; height:{size}px;"></div>'
stars_html += '</div>'
st.markdown(stars_html, unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero">
    <h1>✨ Welcome to KK's Stock Predictor ✨</h1>
    <p>Use deep learning magic 🪄 to forecast tomorrow’s stock prices with our glowing GRU-powered wizardry.  
    Choose a stock, view real predictions, and retrain whenever you want!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>KK's Stock Predictor</h1>", unsafe_allow_html=True)

tickers = ["RELIANCE", "TCS", "COLPAL", "INFY", "PAGEIND", "ITC"]

# Initialize selected stock in session state
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = None

st.markdown("<h3 class='subtitle'> ✦ Tap a stock to reveal its magic </h3>", unsafe_allow_html=True)

# Glowing buttons
st.markdown('<div class="button-grid">', unsafe_allow_html=True)
for ticker in tickers:
    if st.button(f"{ticker}", key=f"select_{ticker}"):
        st.session_state.selected_stock = ticker
st.markdown('</div>', unsafe_allow_html=True)

selected_stock = st.session_state.selected_stock

# Guard clause
if selected_stock is None:
    st.warning("👈 Click a stock card above to begin!")
    st.stop()

# Show last trained time
txt_file = f"last_trained_{selected_stock}.txt"
if os.path.exists(txt_file):
    with open(txt_file) as f:
        last_time = f.read()
    st.caption(f"🕓 Last retrained on: {last_time}")
else:
    st.caption("🕓 Model has not been retrained yet.")

# Retrain button (before predictions)
retrained = False
if st.button("🔁 Retrain Model"):
    with st.spinner(f"Retraining {selected_stock} model..."):
        train_and_save_model(selected_stock)
    st.success(f"{selected_stock} model retrained and saved!")
    retrained = True

# Run GRU model
with st.spinner(f"Running GRU model for {selected_stock}..."):
    actual, predicted, rmse, r2, next_price = get_predictions(selected_stock)

# Plot results
st.markdown("### 📉 Test Set: Actual vs Predicted")
fig = go.Figure()
fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual'))
fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted'))
fig.update_layout(
    plot_bgcolor='#0f0f1a',
    paper_bgcolor='#0f0f1a',
    font=dict(color='#ffffff'),
)
st.plotly_chart(fig, use_container_width=True)

# Prediction card
st.markdown(f"""
<div class='prediction-card'>
    <h2>🔮 Tomorrow's Predicted Closing Price: <span>₹{next_price:.2f}</span></h2>
    <p>📊 RMSE: {rmse:.2f} | R² Score: {r2:.2f}</p>
</div>
""", unsafe_allow_html=True)
