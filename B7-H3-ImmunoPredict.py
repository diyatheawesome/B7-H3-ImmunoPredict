"""
B7-H3 ImmunoPredict: Clinical Decision Support Tool
Streamlit app for predicting B7-H3 expression in pediatric brain tumors
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import plotly.graph_objects as go

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="B7-H3 ImmunoPredict",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================================
# LOAD MODELS & CONFIGURATION
# ========================================
def load_models():
    """Load trained models and configuration from the same folder as app.py"""
    try:
        # Load models
        with open("b7h3_model_MB.pkl", "rb") as f:
            model_mb = pickle.load(f)
        with open("b7h3_model_DMG.pkl", "rb") as f:
            model_dmg = pickle.load(f)
        with open("model_config.pkl", "rb") as f:
            config = pickle.load(f)

        return model_mb, model_dmg, config, None

    except FileNotFoundError as e:
        return None, None, None, f"Model files not found: {e}"
    except Exception as e:
        return None, None, None, f"Error loading models: {e}"


# Load models
model_mb, model_dmg, config, error = load_models()

# Stop if models failed to load
if config is None:
    st.error(f"⚠️ {error}")
    st.info("""
    **Setup Instructions:**
    1. Make sure these files exist in the same folder as app.py:
       - b7h3_model_MB.pkl
       - b7h3_model_DMG.pkl
       - model_config.pkl
    2. Filenames must match exactly (no extra spaces or `(1)` in the name)
    """)
    st.stop()


# ========================================
# CUSTOM CSS
# ========================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .gene-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
    }
    .high-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ff8787 100%);
        color: white;
    }
    .low-prediction {
        background: linear-gradient(135deg, #51cf66 0%, #69db7c 100%);
        color: white;
    }
    .recommendation-box {
        background-color: #e7f5ff;
        border-left: 4px solid #1c7ed6;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# HEADER
# ========================================
col1, col2 = st.columns([1, 12])
with col1:
    st.markdown("# 🧬")
with col2:
    st.markdown('<p class="main-header">B7-H3 ImmunoPredict</p>', unsafe_allow_html=True)

st.markdown('<p class="sub-header">Predicting Immunotherapy Response in Pediatric Brain Tumors</p>', unsafe_allow_html=True)

st.markdown("""
This clinical decision support tool predicts **B7-H3 expression** based on key RNA signatures 
in **Diffuse Midline Glioma (DMG)** and **Medulloblastoma (MB)**.
""")

# Check if models loaded successfully
if error:
    st.error(f"⚠️ {error}")
    st.info("""
    **Setup Instructions:**
    1. Create a `models/` folder in your app directory
    2. Place these files in the folder:
       - `b7h3_model_MB.pkl`
       - `b7h3_model_DMG.pkl`
       - `model_config.pkl`
    """)
    st.stop()

st.markdown("---")

# ========================================
# TUMOR TYPE SELECTION
# ========================================
st.markdown("## 🎯 Step 1: Select Tumor Type")

tumor_type = st.selectbox(
    "Select the tumor type:",
    options=["Medulloblastoma (MB)", "Diffuse Midline Glioma (DMG)"],
    index=0,
    help="Choose the pediatric brain tumor type for prediction"
)

tumor_key = "MB" if "Medulloblastoma" in tumor_type else "DMG"
model = model_mb if tumor_key == "MB" else model_dmg
tumor_config = config[tumor_key]

# Display model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Accuracy", f"{tumor_config['accuracy']:.1%}")
with col2:
    st.metric("ROC-AUC Score", f"{tumor_config['roc_auc']:.3f}")
with col3:
    st.metric("Features Used", len(tumor_config['genes']))

# ========================================
# GENE INFORMATION
# ========================================
st.markdown("## 📊 Step 2: Review Key Predictors")

st.info(f"""
**{tumor_key} Model** uses the following **6 gene signatures** to predict B7-H3 expression:
- **High B7-H3 Predictors:** {', '.join(tumor_config['high_predictors'])}
- **Low B7-H3 Predictors:** {', '.join(tumor_config['low_predictors'])}
""")

with st.expander("ℹ️ What are Z-scores?"):
    st.markdown("""
    **Z-scores** represent normalized gene expression values:
    - **Positive values** (e.g., +0.5, +1.0): Gene expression is **above average**
    - **Zero (0.0)**: Gene expression is at the **population average**
    - **Negative values** (e.g., -0.5, -1.0): Gene expression is **below average**
    
    Typical range: -3.0 to +3.0
    """)

# ========================================
# GENE Z-SCORE INPUTS
# ========================================
st.markdown("## 🧪 Step 3: Enter Gene Z-scores")

st.markdown("Enter the normalized Z-scores for each gene from the patient's RNA sequencing data:")

# Initialize session state for gene values
if 'gene_values' not in st.session_state:
    st.session_state.gene_values = {}

gene_values = {}

# HIGH Predictors
st.markdown("### 🔺 High B7-H3 Predictors")
st.markdown("*Higher expression of these genes correlates with HIGH B7-H3*")

cols = st.columns(3)
for idx, gene in enumerate(tumor_config['high_predictors']):
    with cols[idx]:
        gene_values[gene] = st.number_input(
            f"**{gene}**",
            value=0.50,
            min_value=-3.0,
            max_value=3.0,
            step=0.01,
            format="%.2f",
            help=f"Enter Z-score for {gene}",
            key=f"{gene}_{tumor_key}"
        )

st.markdown("---")

# LOW Predictors
st.markdown("### 🔻 Low B7-H3 Predictors")
st.markdown("*Higher expression of these genes correlates with LOW B7-H3*")

cols = st.columns(3)
for idx, gene in enumerate(tumor_config['low_predictors']):
    with cols[idx]:
        gene_values[gene] = st.number_input(
            f"**{gene}**",
            value=-0.50,
            min_value=-3.0,
            max_value=3.0,
            step=0.01,
            format="%.2f",
            help=f"Enter Z-score for {gene}",
            key=f"{gene}_{tumor_key}"
        )

# ========================================
# PREDICTION
# ========================================
st.markdown("---")
st.markdown("## 🔬 Step 4: Generate Prediction")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Predict B7-H3 Expression", type="primary", use_container_width=True)

if predict_button:
    try:
        # Prepare input in correct order
        X = np.array([[gene_values[gene] for gene in tumor_config['genes']]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        prob_low = probability[0]
        prob_high = probability[1]
        
        # ========================================
        # DISPLAY RESULTS
        # ========================================
        st.markdown("---")
        st.markdown("# 📊 Prediction Results")
        
        # Prediction box
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box high-prediction">
                <h1>⚠️ HIGH B7-H3 Expression</h1>
                <h2>Confidence: {prob_high:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box low-prediction">
                <h1>✅ LOW B7-H3 Expression</h1>
                <h2>Confidence: {prob_low:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Probability visualization
        st.markdown("### Prediction Confidence")
        fig = go.Figure(data=[
            go.Bar(
                x=['Low B7-H3', 'High B7-H3'],
                y=[prob_low * 100, prob_high * 100],
                marker_color=['#51cf66', '#ff6b6b'],
                text=[f'{prob_low:.1%}', f'{prob_high:.1%}'],
                textposition='auto',
            )
        ])
        fig.update_layout(
            yaxis_title="Probability (%)",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ========================================
        # CLINICAL RECOMMENDATIONS
        # ========================================
        st.markdown("---")
        st.markdown("## 💊 Clinical Recommendations")
        
        if prediction == 1:  # HIGH B7-H3
            st.markdown("""
            <div class="recommendation-box">
                <h3>🎯 Recommended: CAR-T Cell Therapy Targeting B7-H3</h3>
                <p><strong>Rationale:</strong> High B7-H3 expression makes this tumor an excellent candidate 
                for B7-H3-targeted CAR-T cell immunotherapy.</p>
                <p><strong>Considerations:</strong></p>
                <ul>
                    <li>Assess patient eligibility for CAR-T therapy</li>
                    <li>Evaluate tumor accessibility and blood-brain barrier considerations</li>
                    <li>Consider combination with checkpoint inhibitors</li>
                    <li>Monitor for cytokine release syndrome (CRS) and neurotoxicity</li>
                </ul>
                <p><strong>Clinical Trials:</strong> Search for ongoing B7-H3 CAR-T trials for pediatric brain tumors</p>
            </div>
            """, unsafe_allow_html=True)
        else:  # LOW B7-H3
            st.markdown("""
            <div class="recommendation-box">
                <h3>🔄 Alternative Immunotherapy Approaches</h3>
                <p><strong>Rationale:</strong> Low B7-H3 expression suggests limited benefit from 
                B7-H3-targeted therapies. Consider alternative immunotherapy strategies.</p>
                <p><strong>Alternative Options:</strong></p>
                <ul>
                    <li><strong>Checkpoint Inhibitors:</strong> Anti-PD-1/PD-L1, Anti-CTLA-4</li>
                    <li><strong>Other CAR-T Targets:</strong> GD2, IL13Rα2, EGFRvIII</li>
                    <li><strong>Oncolytic Virotherapy:</strong> Engineered viruses to target tumor cells</li>
                    <li><strong>Vaccine Therapy:</strong> Personalized neoantigen vaccines</li>
                    <li><strong>Combination Therapy:</strong> Immunotherapy + radiation/chemotherapy</li>
                </ul>
                <p><strong>Next Steps:</strong> Profile additional immune markers to guide therapy selection</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ========================================
        # INPUT SUMMARY
        # ========================================
        st.markdown("---")
        with st.expander("📋 View Input Summary"):
            input_df = pd.DataFrame({
                'Gene': tumor_config['genes'],
                'Z-score': [gene_values[gene] for gene in tumor_config['genes']],
                'Category': ['High Predictor' if gene in tumor_config['high_predictors'] else 'Low Predictor' 
                            for gene in tumor_config['genes']]
            })
            st.dataframe(input_df, use_container_width=True)
            
            # Export option
            csv = input_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Input Data (CSV)",
                data=csv,
                file_name=f"b7h3_prediction_{tumor_key}.csv",
                mime="text/csv"
            )
        
    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
        st.exception(e)

# ========================================
# FOOTER
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em; padding: 2rem 0;'>
    <p>⚠️ <strong>RESEARCH USE ONLY</strong></p>
    <p>This tool is for research purposes only and should NOT be used as the sole basis for clinical decision-making.</p>
    <p>Always consult with qualified healthcare professionals and consider comprehensive clinical evaluation.</p>
    <p style='margin-top: 1rem;'>Developed for WSSEF Project | Based on RNA sequencing data from pediatric brain tumor cohorts</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with additional info
with st.sidebar:
    st.markdown("### 📚 About")
    st.markdown(f"""
    **Current Model:** {tumor_type}
    
    **Performance Metrics:**
    - Accuracy: {tumor_config['accuracy']:.1%}
    - ROC-AUC: {tumor_config['roc_auc']:.3f}
    
    **Model Details:**
    - Algorithm: Random Forest
    - Features: {len(tumor_config['genes'])} genes
    - Training samples: Based on normalized RNA-seq data
    """)
    
    st.markdown("---")
    st.markdown("### ❓ Help")
    st.markdown("""
    **How to use:**
    1. Select tumor type
    2. Enter gene Z-scores
    3. Click 'Predict'
    4. Review recommendations
    
    **Need Help?**
    Contact the research team for questions about gene expression data or interpretation.
    """)
