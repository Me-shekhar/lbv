import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from prediction_model import PredictionModel
from styles import apply_custom_styles
import os

# Configure page
st.set_page_config(
    page_title="LBV Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom styles
apply_custom_styles()

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        # Try to load from the uploaded file path
        csv_path = "DataVizHub/attached_assets/final_dataset_after_preprocessing (2.0).csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
        else:
            # Fallback to looking for any CSV file in the directory
            csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
            if csv_files:
                df = pd.read_csv(csv_files[0])
            else:
                st.error("Dataset file not found. Please ensure the CSV file is available.")
                return None
        
        # Process the data
        processor = DataProcessor()
        processed_df = processor.preprocess_data(df)
        return processed_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_model(data):
    """Load and cache the prediction model"""
    if data is None:
        return None
    
    model = PredictionModel()
    model.train(data)
    return model

def main():
    # Sidebar menu
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>📋 Menu</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # About Section
        with st.expander("ℹ️ About", expanded=False):
            # Load model info
            data = load_data()
            model = load_model(data) if data is not None else None
            
            if model and hasattr(model, 'get_model_info'):
                model_info = model.get_model_info()
                if model_info:
                    
                    st.markdown(f"**Hydrocarbons Supported:** {model_info.get('available_hydrocarbons', 0)}")
                    
                    if 'training_scores' in model_info and model_info['training_scores']:
                        best_model = model_info['model_type']
                        if best_model in model_info['training_scores']:
                            scores = model_info['training_scores'][best_model]
                            r2_score = scores.get('r2_score', 0)
                            rmse = scores.get('rmse', 0)
                            mae = scores.get('mae', 0)
                            
                            st.markdown(f"**R² Score:** {r2_score:.3f}")
                            st.markdown(f"**RMSE:** {rmse:.2f} cm/s")
                            st.markdown(f"**MAE:** {mae:.2f} cm/s")
            else:
                st.markdown("### Model Information")
                st.markdown("**Algorithm:** Random Forest Regressor")
                st.markdown("**Purpose:** Laminar Burning Velocity Prediction")
            
            st.markdown("---")
            st.markdown("### Why Random Forest?")
            st.markdown("**Random Forest** was selected because:")
            st.markdown("• **High Accuracy:** Excellent performance on non-linear combustion data")
            st.markdown("• **Robustness:** Handles multiple hydrocarbon types effectively")
            st.markdown("• **Feature Importance:** Identifies key parameters affecting LBV")
            st.markdown("• **Overfitting Resistance:** Ensemble method reduces variance")
            st.markdown("• **Interpretability:** Provides insights into combustion physics")
        
        # Developers Section
        with st.expander("👥 Developers", expanded=False):
            st.markdown("### Development Team")
            st.markdown("**Final Year Mechanical Engineering Students**")
            st.markdown("**Pimpri Chinchwad College of Engineering, Ravet**")
            st.markdown("**Pune, Maharashtra**")
            st.markdown("---")
            
            st.markdown("**Shekhar Sonar**")
            st.markdown("📧 sonarshekhar641@gmail.com")
            st.markdown("")
            
            st.markdown("**Sujal Fiske**")
            st.markdown("📧 sujal.fiske_mech22@pccoer.in")
            st.markdown("")
            
            st.markdown("**Karan Shinde**")
            st.markdown("📧 karan.shinde_mech23@pccoer.in")
        
        # Mentor/Project Guide Section
        with st.expander("👨‍🏫 Mentor / Project Guide", expanded=False):
            st.markdown("### Project Supervisor")
            st.markdown("")
            
            st.markdown("**Shawnam**")
            st.markdown("📧 shawnam.ae111@gmail.com")
            st.markdown("🏛️ Department of Aerospace Engineering")
            st.markdown("🎓 Indian Institute of Technology Bombay")
            st.markdown("📍 Mumbai 400076, India")
        
        # Resources Section
        with st.expander("📚 Resources", expanded=False):
            st.markdown("### References")
            
            st.markdown("**Textbook:**")
            st.markdown("Turns, S. R., 2020, *An Introduction to Combustion: Concepts and Applications*, McGraw-Hill Education.")
            st.markdown("")
            
            st.markdown("**Key Research Papers:**")
            st.markdown("---")
            
            st.markdown("1. *Laminar burning velocity measurements of ethyl valerate-air flames at elevated temperatures with mechanism modifications*")
            st.markdown("**Authors:** Shawnam, Rohit Kumar, E.V. Jithin, Ratna Kishore Velamati, Sudarshan Kumar")
            st.markdown("")
            
            st.markdown("2. *Experimental measurements of laminar burning velocity of premixed propane-air flames at higher pressure and temperature conditions*")
            st.markdown("**Authors:** Vijay Shinde, Amardeep Fulzele, Sudarshan Kumar")
            st.markdown("")
            
            st.markdown("3. *Laminar burning velocity measurements of NH3/H2+Air mixtures at elevated temperatures*")
            st.markdown("**Authors:** Shawnam, Pragya Berwal, Muskaan Singh, Sudarshan Kumar")
            st.markdown("")
            
            st.markdown("4. *Combustion of N-Decane+air Mixtures To Investigate Laminar Burning Velocity Measurements At Elevated Temperatures*")
            st.markdown("**Authors:** Rohit Kumar, Ratna Kishore Velamati, Sudarshan Kumar")
            st.markdown("")
            
            st.markdown("**Additional Sources:**")
            st.markdown("+ 30 more research papers on laminar burning velocity measurements")
            st.markdown("+ Various hydrocarbon combustion studies")
            st.markdown("+ High-temperature flame propagation research")

    # Header with hamburger menu
    st.markdown("""
    <div class="header">
        <div class="header-content">
            <div class="header-icon">🔥</div>
            <h1>LBV Predictor</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Load model
    model = load_model(data)
    if model is None:
        st.error("Failed to load prediction model")
        st.stop()
    
    # Main prediction section
    st.markdown("""
    <div class="prediction-section">
        <div class="section-header">
            <span class="section-icon">🔥</span>
            <h2>Laminar Burning Velocity Prediction</h2>
        </div>
        <p class="section-description">
            Predict the Laminar Burning Velocity (LBV) for different hydrocarbons 
            based on temperature, equivalence ratio, and pressure.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get unique hydrocarbons
    hydrocarbons = sorted(data['Hydrocarbon'].unique())
    
    # Hydrocarbon selection (full width)
    st.markdown("<div class='input-group'>", unsafe_allow_html=True)
    st.markdown("<label class='input-label'>Hydrocarbon</label>", unsafe_allow_html=True)
    selected_hydrocarbon = st.selectbox(
        "Hydrocarbon",
        options=hydrocarbons,
        index=hydrocarbons.index("DME - air mixture") if "DME - air mixture" in hydrocarbons else 0,
        key="hydrocarbon",
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input form - 3 equal columns
    col1, col2, col3 = st.columns(3)
    
    # Get ranges for selected hydrocarbon
    hydrocarbon_data = data[data['Hydrocarbon'] == selected_hydrocarbon]
    temp_min = float(hydrocarbon_data['Ti (K)'].min())
    temp_max = float(hydrocarbon_data['Ti (K)'].max())
    ratio_min = float(hydrocarbon_data['equivalent ratio'].min())
    ratio_max = float(hydrocarbon_data['equivalent ratio'].max())
    pressure_min = float(hydrocarbon_data['Pressure (atm)'].min())
    pressure_max = float(hydrocarbon_data['Pressure (atm)'].max())
    
    with col1:
        st.markdown("<div class='input-group'>", unsafe_allow_html=True)
        st.markdown("<label class='input-label'>Initial Temperature (K)</label>", unsafe_allow_html=True)
        temperature = st.number_input(
            "Initial Temperature (K)",
            min_value=temp_min,
            max_value=temp_max,
            value=450.0,
            step=1.0,
            key="temperature",
            label_visibility="collapsed"
        )
        st.markdown(f"<small class='range-info'>Valid range: {temp_min:.1f} - {temp_max:.1f} K</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='input-group'>", unsafe_allow_html=True)
        st.markdown("<label class='input-label'>Equivalence Ratio (φ)</label>", unsafe_allow_html=True)
        equivalence_ratio = st.number_input(
            "Equivalence Ratio (φ)",
            min_value=ratio_min,
            max_value=ratio_max,
            value=0.7,
            step=0.01,
            key="ratio",
            label_visibility="collapsed"
        )
        st.markdown(f"<small class='range-info'>Valid range: {ratio_min:.2f} - {ratio_max:.2f}</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='input-group'>", unsafe_allow_html=True)
        st.markdown("<label class='input-label'>Pressure (atm)</label>", unsafe_allow_html=True)
        pressure = st.number_input(
            "Pressure (atm)",
            min_value=pressure_min,
            max_value=pressure_max,
            value=1.0,
            step=0.1,
            key="pressure",
            label_visibility="collapsed"
        )
        st.markdown(f"<small class='range-info'>Valid range: {pressure_min:.1f} - {pressure_max:.1f} atm</small>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Unit toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        unit_conversion = st.toggle("Convert to m/s", key="unit_toggle")
    
    # Predict button
    if st.button("🔥 Predict LBV", key="predict_btn", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(selected_hydrocarbon, temperature, equivalence_ratio, pressure)
            
            if prediction is not None:
                # Convert units if requested
                if unit_conversion:
                    prediction_value = prediction / 100  # cm/s to m/s
                    unit = "m/s"
                else:
                    prediction_value = prediction
                    unit = "cm/s"
                
                # Results section
                st.markdown("""
                <div class="results-section">
                    <div class="results-header">
                        <span class="results-icon">📊</span>
                        <h2>Prediction Results</h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Results display
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    <div class="results-card">
                        <h3>Input Parameters</h3>
                        <div class="parameter-list">
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="parameter-item">
                            <span class="param-label">Hydrocarbon</span>
                            <span class="param-value">{selected_hydrocarbon}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="param-label">Temperature</span>
                            <span class="param-value">{temperature} K</span>
                        </div>
                        <div class="parameter-item">
                            <span class="param-label">Equivalence Ratio</span>
                            <span class="param-value">{equivalence_ratio}</span>
                        </div>
                        <div class="parameter-item">
                            <span class="param-label">Pressure</span>
                            <span class="param-value">{pressure} atm</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="results-card">
                        <h3>Predicted LBV</h3>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                        <div class="prediction-value">
                            <span class="value">{prediction_value:.1f}</span>
                            <span class="unit">{unit}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                
            else:
                st.error("Unable to make prediction with the given parameters. Please check your inputs.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    main()
