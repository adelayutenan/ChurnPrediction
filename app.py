import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from PIL import Image
import time

# Configure page
st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

model = load_model()

# =============================================
# ============ SIDEBAR COMPONENTS =============
# =============================================

# Sidebar header with logo
st.sidebar.markdown("""
<div class="sidebar-header">
    <div class="logo-container">
        <img src="https://cdn-icons-png.flaticon.com/512/2285/2285533.png" class="logo">
        <h2>Customer Profile</h2>
    </div>
    <p class="subtitle">Enter customer details</p>
</div>
""", unsafe_allow_html=True)

# Customer input form
with st.sidebar.expander("🧑‍💼 DEMOGRAPHICS", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        SeniorCitizen = st.selectbox("Age Group", ["Under 60", "60+"])
    with col2:
        Partner = st.selectbox("Partner", ["No", "Yes"])
    
    col1, col2 = st.columns(2)
    with col1:
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
    with col2:
        tenure_group = st.selectbox("Tenure", ['1-12 mo', '13-24 mo', '25-36 mo', '37-48 mo', '49-60 mo', '61-72 mo'])

with st.sidebar.expander("📞 PHONE SERVICES", expanded=True):
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    if PhoneService == "Yes":
        MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes"])
    else:
        MultipleLines = "No phone service"

with st.sidebar.expander("🌐 INTERNET SERVICES", expanded=True):
    InternetService = st.selectbox("Internet Type", ["DSL", "Fiber optic", "None"])
    
    if InternetService != "None":
        cols = st.columns(2)
        with cols[0]:
            OnlineSecurity = st.selectbox("Security", ["No", "Yes"])
            OnlineBackup = st.selectbox("Backup", ["No", "Yes"])
            StreamingTV = st.selectbox("Stream TV", ["No", "Yes"])
        with cols[1]:
            DeviceProtection = st.selectbox("Device Protection", ["No", "Yes"])
            TechSupport = st.selectbox("Tech Support", ["No", "Yes"])
            StreamingMovies = st.selectbox("Stream Movies", ["No", "Yes"])
    else:
        OnlineSecurity = OnlineBackup = DeviceProtection = TechSupport = StreamingTV = StreamingMovies = "No internet service"

with st.sidebar.expander("💳 ACCOUNT DETAILS", expanded=True):
    Contract = st.selectbox("Contract", ["Monthly", "1 Year", "2 Years"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic Check", 
        "Mailed Check", 
        "Bank Transfer", 
        "Credit Card"
    ])
    
    MonthlyCharges = st.slider("Monthly Charges ($)", 20, 120, 70)
    TotalCharges = st.slider("Total Charges ($)", 100, 10000, 2000)

# =============================================
# ============= MAIN DASHBOARD ================
# =============================================

st.markdown("""
<div class="dashboard-header">
    <div class="header-content">
        <div class="header-text">
            <h1>TELCO BUSINESS<br>CHURN PREDICTION</h1>
            <p class="subheader">Customer retention analytics dashboard</p>
        </div>
        <div class="header-image">
            <img src="https://blog.usetada.com/hs-fs/hubfs/Customer%20Churn%20and%20How%20to%20Stop%20It.png?width=2400&name=Customer%20Churn%20and%20How%20to%20Stop%20It.png" alt="Company Logo" style="max-height: 350px;">
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key metrics row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown("""
    <div class="metric-card" style="background-color: #e3f2fd">
        <div class="metric-value">85.7%</div>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric-card" style="background-color: #e8f5e9">
        <div class="metric-value">23.5%</div>
        <div class="metric-label">Churn Rate</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="metric-card" style="background-color: #fff3e0">
        <div class="metric-value">$1,240</div>
        <div class="metric-label">Avg. Value</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric-card" style="background-color: #f3e5f5">
        <div class="metric-value">142</div>
        <div class="metric-label">At-Risk</div>
    </div>
    """, unsafe_allow_html=True)

# Main content columns
col1, col2 = st.columns([2, 1])

with col1:
    # Prediction results section
    st.markdown("""
    <div class="section-header">
        <h2>📊 Churn Risk Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Analyze Customer", type="primary", use_container_width=True):
        with st.spinner('Analyzing customer data...'):
            time.sleep(1)  # Simulate processing
            
            # Prepare data
            data = {
                'SeniorCitizen': 1 if SeniorCitizen == "60+" else 0,
                'Partner': Partner,
                'Dependents': Dependents,
                'PhoneService': PhoneService,
                'MultipleLines': MultipleLines,
                'InternetService': InternetService,
                'OnlineSecurity': OnlineSecurity,
                'OnlineBackup': OnlineBackup,
                'DeviceProtection': DeviceProtection,
                'TechSupport': TechSupport,
                'StreamingTV': StreamingTV,
                'StreamingMovies': StreamingMovies,
                'Contract': Contract,
                'PaperlessBilling': PaperlessBilling,
                'PaymentMethod': PaymentMethod,
                'MonthlyCharges': MonthlyCharges,
                'TotalCharges': TotalCharges,
                'tenure_group': tenure_group
            }
            
            input_df = pd.DataFrame([data])
            input_encoded = pd.get_dummies(input_df)
            
            # Ensure all features match the model's training features
            model_features = model.get_booster().feature_names
            for col in model_features:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[model_features]
            
            # Make prediction
            prediction = model.predict(input_encoded)
            proba = model.predict_proba(input_encoded)
            churn_prob = proba[0][1]*100
            
            # Display results
            result_container = st.container()
            with result_container:
                result_col1, result_col2 = st.columns([1, 2])
                
                with result_col1:
                    # Gauge chart with pastel colors
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = churn_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Probability", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#7986cb"},
                            'bar': {'color': "#7986cb"},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': "#aed581"},
                                {'range': [30, 70], 'color': "#ffd54f"},
                                {'range': [70, 100], 'color': "#ff8a65"}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': churn_prob}
                        }
                    ))
                    
                    fig.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=50, b=10),
                        font={'family': "Arial"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with result_col2:
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="alert-card" style="background-color: #ffebee">
                            <h3>🚨 High Churn Risk: {churn_prob:.1f}%</h3>
                            <p>This customer is likely to cancel service</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="alert-card" style="background-color: #e8f5e9">
                            <h3>✅ Low Churn Risk: {churn_prob:.1f}%</h3>
                            <p>This customer is likely to stay</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Key factors
                    st.markdown("""
                    <div class="factors-card" style="background-color: #e3f2fd">
                        <h4>🔍 Key Factors</h4>
                        <div class="factors-grid">
                            <div class="factor-item">
                                <div class="factor-label">Contract</div>
                                <div class="factor-value">{}</div>
                            </div>
                            <div class="factor-item">
                                <div class="factor-label">Monthly</div>
                                <div class="factor-value">${}</div>
                            </div>
                            <div class="factor-item">
                                <div class="factor-label">Payment</div>
                                <div class="factor-value">{}</div>
                            </div>
                            <div class="factor-item">
                                <div class="factor-label">Internet</div>
                                <div class="factor-value">{}</div>
                            </div>
                        </div>
                    </div>
                    """.format(
                        Contract,
                        MonthlyCharges,
                        PaymentMethod,
                        InternetService
                    ), unsafe_allow_html=True)
            
            # Recommendations section
            st.markdown("""
            <div class="section-header">
                <h2>💡 Recommendations</h2>
            </div>
            """, unsafe_allow_html=True)
            
            rec_col1, rec_col2 = st.columns(2)
            
            with rec_col1:
                st.markdown("""
                <div class="recommendation-card" style="background-color: #fff8e1">
                    <h4>🛡️ Retention Strategies</h4>
                    <ul>
                        <li>Personalized retention call</li>
                        <li>15-20% discount offer</li>
                        <li>Free service upgrade</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with rec_col2:
                st.markdown("""
                <div class="recommendation-card" style="background-color: #f1f8e9">
                    <h4>📈 Growth Opportunities</h4>
                    <ul>
                        <li>Upsell premium features</li>
                        <li>Referral program</li>
                        <li>Family plan options</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

with col2:
    # Customer snapshot
    st.markdown("""
    <div class="section-header">
        <h2>👤 Customer Profile</h2>
    </div>
    <div class="profile-card" style="background-color: #f5f5f5">
        <div class="profile-header">
            <div class="avatar">👩‍💼</div>
            <div class="profile-info">
                <h3>Sample Customer</h3>
                <p>Telco Subscriber</p>
            </div>
        </div>
        <div class="profile-details">
            <div class="detail-item">
                <span class="detail-label">Contract:</span>
                <span class="detail-value">{}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Tenure:</span>
                <span class="detail-value">{}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Monthly:</span>
                <span class="detail-value">${}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Total Spent:</span>
                <span class="detail-value">${:,.0f}</span>
            </div>
        </div>
    </div>
    """.format(
        Contract,
        tenure_group,
        MonthlyCharges,
        TotalCharges
    ), unsafe_allow_html=True)
    
    # Tips card
    st.markdown("""
    <div class="section-header">
        <h2>💎 Pro Tips</h2>
    </div>
    <div class="trend-card" style="background-color: #f5f5f5">
        <img src="https://arsigakonics.org/wp-content/uploads/2024/05/Strategies-for-Maximizing-Employee-Potential-through-Corporate-Training-Image.png">
    </div>
    <div class="tips-card" style="background-color: #e3f2fd">
        <div class="tip-item">
            <div class="tip-icon">💡</div>
            <div class="tip-content">Customers with longer contracts churn less</div>
        </div>
        <div class="tip-item">
            <div class="tip-icon">💡</div>
            <div class="tip-content">Tech support reduces churn by 18%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2023 Telco Analytics | Customer Retention Platform</p>
</div>
""", unsafe_allow_html=True)