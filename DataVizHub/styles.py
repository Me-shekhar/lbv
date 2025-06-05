import streamlit as st

def apply_custom_styles():
    """Apply custom CSS styles to match the exact UI design"""
    
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #1F2937 0%, #111827 100%);
        color: #F9FAFB;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Styles */
    .header {
        background: linear-gradient(135deg, #6B46C1 0%, #553C9A 100%);
        padding: 1.5rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 1rem 1rem;
        box-shadow: 0 4px 20px rgba(107, 70, 193, 0.3);
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .header-icon {
        font-size: 2rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .header h1 {
        color: white;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Prediction Section */
    .prediction-section {
        background: rgba(55, 65, 81, 0.5);
        border: 1px solid rgba(75, 85, 99, 0.3);
        border-radius: 1rem;
        padding: 2rem;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
    }
    
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
    }
    
    .section-header h2 {
        color: #F9FAFB;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }
    
    .section-description {
        color: #D1D5DB;
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Input Styles */
    .input-group {
        margin-bottom: 1.5rem;
    }
    
    .input-label {
        display: block;
        color: #F9FAFB;
        font-weight: 500;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    
    .range-info {
        color: #9CA3AF;
        font-size: 0.75rem;
        margin-top: 0.25rem;
        display: block;
    }
    
    /* Streamlit Input Overrides */
    .stSelectbox > div > div {
        background-color: rgba(55, 65, 81, 0.8) !important;
        border: 1px solid rgba(107, 70, 193, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #F9FAFB !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #6B46C1 !important;
        box-shadow: 0 0 0 2px rgba(107, 70, 193, 0.2) !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: rgba(55, 65, 81, 0.8) !important;
        border: 1px solid rgba(107, 70, 193, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #F9FAFB !important;
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #6B46C1 !important;
        box-shadow: 0 0 0 2px rgba(107, 70, 193, 0.2) !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #6B46C1 0%, #553C9A 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.75rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(107, 70, 193, 0.3) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(107, 70, 193, 0.4) !important;
    }
    
    /* Toggle Styles */
    .stToggle > div {
        justify-content: flex-end !important;
    }
    
    /* Results Section */
    .results-section {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 1rem;
        padding: 1.5rem 2rem;
        margin: 2rem 0 1rem 0;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }
    
    .results-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .results-icon {
        font-size: 1.5rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .results-header h2 {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Results Cards */
    .results-card {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin-top: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .results-card h3 {
        color: white;
        font-size: 1.125rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
    }
    
    /* Parameter List */
    .parameter-list {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }
    
    .parameter-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .parameter-item:last-child {
        border-bottom: none;
    }
    
    .param-label {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    .param-value {
        color: white;
        font-weight: 600;
    }
    
    /* Prediction Value */
    .prediction-value {
        text-align: center;
        padding: 2rem 1rem;
    }
    
    .prediction-value .value {
        display: block;
        font-size: 3rem;
        font-weight: 700;
        color: #32CD32;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0.5rem;
    }
    
    .prediction-value .unit {
        display: block;
        font-size: 1.25rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header {
            padding: 1rem;
        }
        
        .prediction-section {
            padding: 1.5rem;
        }
        
        .prediction-value .value {
            font-size: 2.5rem;
        }
        
        .results-card {
            padding: 1rem;
        }
    }
    
    /* Error Messages */
    .stAlert > div {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #FCA5A5 !important;
    }
    
    /* Success Messages */
    .stSuccess > div {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #6EE7B7 !important;
    }
    
    /* Sidebar Styles */
    .sidebar-header {
        background: linear-gradient(135deg, #6B46C1 0%, #553C9A 100%);
        padding: 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    
    .sidebar-header h2 {
        color: white;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(55, 65, 81, 0.5) !important;
        border: 1px solid rgba(75, 85, 99, 0.3) !important;
        border-radius: 0.5rem !important;
        color: #F9FAFB !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(55, 65, 81, 0.3) !important;
        border: 1px solid rgba(75, 85, 99, 0.3) !important;
        border-top: none !important;
        border-radius: 0 0 0.5rem 0.5rem !important;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Column spacing */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }
    
    /* Remove default selectbox styling */
    .stSelectbox label {
        display: none !important;
    }
    
    .stNumberInput label {
        display: none !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(55, 65, 81, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(107, 70, 193, 0.6);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(107, 70, 193, 0.8);
    }
    </style>
    """, unsafe_allow_html=True)
