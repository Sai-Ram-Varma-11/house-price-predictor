import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("house_price_model.joblib")

# -------------------- Custom CSS for futuristic UI --------------------
st.markdown("""
<style>
/* Overall dark background */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
    color: #ffffff;
    background-image: radial-gradient(circle at center, #000000 0%, #111111 100%);
    position: relative;
    overflow: hidden;
}

/* Glassy main container */
[data-testid="stForm"] {
    backdrop-filter: blur(10px) saturate(180%);
    background-color: rgba(20, 20, 30, 0.5);
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #a020f0;
}

/* Headings */
h1 {
    color: #ffffff;
    text-shadow: 0 0 15px #a020f0;
    font-size: 2.5em;
    text-align: center;
    margin-bottom: 30px;
    user-select: none;
}

/* Prediction result styling */
.prediction-container {
    background: rgba(160, 32, 240, 0.1);
    border: 2px solid #a020f0;
    border-radius: 10px;
    padding: 20px;
    margin-top: 30px;
    text-align: center;
}

.prediction-value {
    color: #00ffff;
    font-size: 24px;
    font-weight: bold;
    text-shadow: 0 0 10px #00ffff;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #a020f0 0%, #ff00ff 100%);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    border: none;
    padding: 10px 25px;
    box-shadow: 0 0 10px #a020f0;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    box-shadow: 0 0 20px #ff00ff;
    transform: translateY(-2px);
}

/* Input fields */
[data-testid="stNumberInput"] input {
    background-color: rgba(255, 255, 255, 0.05);
    border: 1px solid #a020f0;
    color: white;
    border-radius: 5px;
}

/* Animated background */
@keyframes float {
    0% { transform: translate(0, 0) scale(1); opacity: 0; }
    50% { opacity: 0.5; }
    100% { transform: translate(var(--tx), var(--ty)) scale(0); opacity: 0; }
}

.floating-ball {
    position: fixed;
    width: 6px;
    height: 6px;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 50%;
    pointer-events: none;
    z-index: -1;
}
</style>

<script>
function createFloatingBalls() {
    const container = document.querySelector('[data-testid="stAppViewContainer"]');
    for (let i = 0; i < 50; i++) {
        const ball = document.createElement('div');
        ball.className = 'floating-ball';
        
        // Random position
        ball.style.left = Math.random() * 100 + 'vw';
        ball.style.top = Math.random() * 100 + 'vh';
        
        // Random animation properties
        ball.style.setProperty('--tx', (Math.random() * 200 - 100) + 'px');
        ball.style.setProperty('--ty', (Math.random() * 200 - 100) + 'px');
        
        // Animation
        ball.style.animation = `float ${3 + Math.random() * 4}s linear infinite`;
        ball.style.animationDelay = Math.random() * 5 + 's';
        
        container.appendChild(ball);
    }
}
// Run after page load
window.addEventListener('load', createFloatingBalls);
</script>
""", unsafe_allow_html=True)

# -------------------- Streamlit App --------------------
st.markdown("<h1>House Price Prediction App</h1>", unsafe_allow_html=True)
st.write("Enter the details below to predict the house price:")

# Create form with glassy styling
with st.form(key="input_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        sqft = st.number_input("Square Footage", min_value=500, max_value=10000, step=50, value=1400)
        bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, step=1, value=3)
    
    with col2:
        bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
    
    submit_button = st.form_submit_button("Predict Price")

# Predict and display result with original styling
if submit_button:
    # Current USD to INR conversion rate (as of 2025)
    USD_TO_INR = 83.5  # You can update this rate periodically
    
    features = np.array([[sqft, bedrooms, bathrooms]])
    prediction_usd = model.predict(features)[0]
    prediction_inr = prediction_usd * USD_TO_INR
    
    # Display prediction in a styled container
    st.markdown(f"""
    <div class="prediction-container">
        <h3>Predicted House Price</h3>
        <div class="prediction-value">₹{prediction_inr:,.2f}</div>
        <div style="font-size: 14px; color: #888; margin-top: 10px;">
            (${prediction_usd:,.2f} USD)
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.5); padding: 20px;'>
    Built with ❤️ using Machine Learning
</div>
""", unsafe_allow_html=True)
