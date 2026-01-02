import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="Consumer Cluster Predictor",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects
@st.cache_resource
def load_model():
    """Load the trained model and preprocessing objects"""
    try:
        with open('model_pipeline.pkl', 'rb') as f:
            model_objects = pickle.load(f)
        return model_objects, None
    except FileNotFoundError:
        error_msg = """
        ⚠️ Model file not found!
        
        Please ensure 'model_pipeline.pkl' is in the same directory as this script.
        
        To generate the model file, run: python claude_export_data_for_powerbi.py
        """
        return None, error_msg
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model_objects, error = load_model()

if error:
    st.error(error)
    st.stop()

# Extract model components
model = model_objects['model']
mlb_encoders = model_objects['mlb_encoders']
ohe_encoder = model_objects['ohe_encoder']
scaler = model_objects['scaler']
feature_columns = model_objects['feature_columns']
multi_response_cols = model_objects['multi_response_cols']
single_response_cols = model_objects['single_response_cols']
top5_likert = model_objects['top5_likert']
cluster_names = model_objects['cluster_names']

# Cluster information
cluster_info = {
    "Mainstream Casual": {
        "description": "Young consumers who buy casual cotton T-shirts, prioritize ease-of-care and pastel aesthetics.",
        "characteristics": [
            "👥 Predominantly young (19-25) and female",
            "👕 Prefer casual cotton T-shirts",
            "🎨 Like pastel colors and simple designs",
            "🛒 Shop primarily online",
            "💰 Budget-conscious, occasional purchasers"
        ],
        "marketing": [
            "Focus on affordable, trendy casual wear",
            "Use social media and online advertising",
            "Emphasize comfort and easy care",
            "Offer frequent promotions"
        ],
        "color": "#4CAF50"
    },
    "Fashion-Forward Modest Consumer": {
        "description": "Female consumers who buy traditional/occasion wear, care about appearance and easy maintenance.",
        "characteristics": [
            "👥 Predominantly female across age groups",
            "👗 Interested in traditional and occasion wear",
            "✨ Value both aesthetics and functionality",
            "🧵 Appreciate quality fabrics (linen, silk)",
            "🛍️ Willing to invest in special pieces"
        ],
        "marketing": [
            "Target with modest fashion collections",
            "Highlight quality and craftsmanship",
            "Use Instagram and fashion influencers",
            "Emphasize ironless and easy-care features"
        ],
        "color": "#E91E63"
    },
    "Traditional Functionalist": {
        "description": "Male consumers who buy formal or semi-formal shirts, prefer durable materials.",
        "characteristics": [
            "👥 Predominantly male, older demographics",
            "👔 Focus on formal and semi-formal wear",
            "💪 Prioritize durability and longevity",
            "🏪 Prefer physical store shopping",
            "📊 Value quality over trends"
        ],
        "marketing": [
            "Emphasize quality and durability",
            "Target professional settings",
            "Use traditional retail channels",
            "Focus on long-term value proposition"
        ],
        "color": "#2196F3"
    }
}

# Prediction functions
def make_prediction(input_data):
    """
    Make prediction based on user inputs
    
    Parameters:
    -----------
    input_data : dict
        Dictionary containing user inputs
        
    Returns:
    --------
    tuple : (predicted_cluster, confidence, probabilities, success, error_message)
    """
    try:
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Preprocess multi-response columns
        mlb_encoded = []
        for col in multi_response_cols:
            input_df[col] = input_df[col].astype(str).apply(
                lambda x: [i.strip() for i in x.split(',') if i.strip()]
            )
            encoded = pd.DataFrame(
                mlb_encoders[col].transform(input_df[col]),
                columns=[f"{col}_{c}" for c in mlb_encoders[col].classes_],
                index=input_df.index
            )
            mlb_encoded.append(encoded)
        
        multi_encoded = pd.concat(mlb_encoded, axis=1) if mlb_encoded else pd.DataFrame(index=input_df.index)
        
        # Single-response encoding
        single_encoded = pd.DataFrame(
            ohe_encoder.transform(input_df[single_response_cols]),
            columns=ohe_encoder.get_feature_names_out(single_response_cols),
            index=input_df.index
        )
        
        # Combine categorical
        cat_encoded = pd.concat([multi_encoded, single_encoded], axis=1)
        
        # Scale Likert
        scaled_likert = pd.DataFrame(
            scaler.transform(input_df[top5_likert]),
            columns=top5_likert,
            index=input_df.index
        )
        
        # Combine all features
        X_final = pd.concat([cat_encoded, scaled_likert], axis=1)
        
        # Add missing columns with zeros
        for col in feature_columns:
            if col not in X_final.columns:
                X_final[col] = 0
        
        # Reorder columns to match training data
        X_final = X_final[feature_columns]
        
        # Make prediction
        prediction = model.predict(X_final)[0]
        probabilities = model.predict_proba(X_final)[0]
        
        predicted_cluster = cluster_names[prediction]
        confidence = probabilities[prediction]
        
        return predicted_cluster, confidence, probabilities, True, None
        
    except Exception as e:
        return None, None, None, False, str(e)


def make_batch_predictions(df):
    """
    Make predictions for a batch of customers from CSV
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe with customer data
        
    Returns:
    --------
    tuple : (results_df, success, error_message)
    """
    try:
        # Validate required columns
        required_cols = single_response_cols + multi_response_cols + top5_likert
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return None, False, f"Missing required columns: {', '.join(missing_cols)}"
        
        # Prepare input data
        X = df[required_cols].copy()
        
        # Preprocess multi-response columns
        mlb_encoded = []
        for col in multi_response_cols:
            X[col] = X[col].astype(str).apply(
                lambda x: [i.strip() for i in x.split(',') if i.strip()]
            )
            encoded = pd.DataFrame(
                mlb_encoders[col].transform(X[col]),
                columns=[f"{col}_{c}" for c in mlb_encoders[col].classes_],
                index=X.index
            )
            mlb_encoded.append(encoded)
        
        multi_encoded = pd.concat(mlb_encoded, axis=1) if mlb_encoded else pd.DataFrame(index=X.index)
        
        # Single-response encoding
        single_encoded = pd.DataFrame(
            ohe_encoder.transform(X[single_response_cols]),
            columns=ohe_encoder.get_feature_names_out(single_response_cols),
            index=X.index
        )
        
        # Combine categorical
        cat_encoded = pd.concat([multi_encoded, single_encoded], axis=1)
        
        # Scale Likert
        scaled_likert = pd.DataFrame(
            scaler.transform(X[top5_likert]),
            columns=top5_likert,
            index=X.index
        )
        
        # Combine all features
        X_final = pd.concat([cat_encoded, scaled_likert], axis=1)
        
        # Add missing columns with zeros
        for col in feature_columns:
            if col not in X_final.columns:
                X_final[col] = 0
        
        # Reorder columns to match training data
        X_final = X_final[feature_columns]
        
        # Make predictions
        predictions = model.predict(X_final)
        probabilities = model.predict_proba(X_final)
        
        # Create results dataframe
        results_df = df.copy()
        results_df['Predicted_Cluster_ID'] = predictions
        results_df['Predicted_Cluster_Name'] = [cluster_names[p] for p in predictions]
        results_df['Prediction_Confidence'] = [probs[pred] for probs, pred in zip(probabilities, predictions)]
        
        # Add probabilities for each cluster
        for i, cluster_name in cluster_names.items():
            results_df[f'Prob_{cluster_name}'] = probabilities[:, i]
        
        return results_df, True, None
        
    except Exception as e:
        return None, False, str(e)


def create_batch_summary(results_df):
    """Create summary statistics for batch predictions"""
    summary = {
        'total_customers': len(results_df),
        'cluster_distribution': results_df['Predicted_Cluster_Name'].value_counts().to_dict(),
        'avg_confidence': results_df['Prediction_Confidence'].mean(),
        'high_confidence_count': len(results_df[results_df['Prediction_Confidence'] >= 0.7]),
        'low_confidence_count': len(results_df[results_df['Prediction_Confidence'] < 0.5])
    }
    return summary


def convert_df_to_excel(df):
    """Convert dataframe to Excel bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    output.seek(0)
    return output

# Main app interface
# Initialize session state for form reset
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0

if 'show_results' not in st.session_state:
    st.session_state.show_results = False

if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

# Header
st.title('🔮 Indonesian Textile Consumer Cluster Predictor')
st.markdown("""
Discover which Indonesian textile consumer cluster you belong to using machine learning. Simply fill in the form below and click **Predict** to see your results!
""")

st.divider()

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Individual Prediction", "🏢 Batch Analysis", "📊 About Clusters", "ℹ️ How It Works"])

# Tab 1: Individual prediction interface
with tab1:
    st.markdown("### Enter Your Information")
    st.caption("Perfect for individual consumers who want to discover their cluster")
    
    # Create two columns for input
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 👤 Demographics")
        
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"],
            help="Select your gender",
            key=f"gender_{st.session_state.form_key}"
        )
        
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Widowed/Divorced"],
            help="Select your current marital status",
            key=f"marital_{st.session_state.form_key}"
        )
        
        age = st.selectbox(
            "Age Group",
            ["19-25", "26-35", "36-45", "45 or above"],
            help="Select your age group",
            key=f"age_{st.session_state.form_key}"
        )
        
        occupation = st.selectbox(
            "Occupation",
            ["Student", "Private office worker", "Government servant", 
             "Freelancer", "Business owner", "Other"],
            help="Select your primary occupation",
            key=f"occupation_{st.session_state.form_key}"
        )
        
        st.markdown("#### 🛍️ Shopping Behavior")
        
        frequency = st.selectbox(
            "How often do you purchase clothing per month?",
            ["Never", "Occasionally (≤1 item)", "Moderate (2–3 items)", "Frequently (≥4 items)"],
            index=1,
            help="Select your typical monthly clothing purchase frequency",
            key=f"frequency_{st.session_state.form_key}"
        )
        
        st.markdown("**Main Purchase Method(s)** *(select all that apply)*")
        purchase_methods = []
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.checkbox("Physical Store", value=False, key=f"phys_{st.session_state.form_key}"):
                purchase_methods.append("Physical Store")
            if st.checkbox("Online platform", value=True, key=f"online_{st.session_state.form_key}"):
                purchase_methods.append("Online platform")
            if st.checkbox("Cross-border platform", value=False, key=f"crossborder_{st.session_state.form_key}"):
                purchase_methods.append("Cross-border platform")
        with col_b:
            if st.checkbox("Social Media", value=False, key=f"social_{st.session_state.form_key}"):
                purchase_methods.append("Social Media")
            if st.checkbox("Other", value=False, key=f"other_{st.session_state.form_key}"):
                purchase_methods.append("Other")
        
        purchase_method_str = ", ".join(purchase_methods) if purchase_methods else "Online platform"
    
    with col2:
        st.markdown("#### ⭐ Your Preferences")
        st.markdown("*Rate your agreement with these statements*")
        st.caption("1 = Strongly Disagree | 5 = Strongly Agree")
        
        st.markdown("**Sustainability & Environmental Concerns:**")
        
        recycling = st.slider(
            "I am likely to participate in recycling programs offered by clothing brands",
            min_value=1,
            max_value=5,
            value=3,
            help="How likely are you to participate in clothing recycling programs?",
            key=f"recycling_{st.session_state.form_key}"
        )
        
        biodegradable = st.slider(
            "I prefer clothing made from biodegradable or recyclable materials",
            min_value=1,
            max_value=5,
            value=3,
            help="Do you prefer environmentally friendly materials?",
            key=f"biodegradable_{st.session_state.form_key}"
        )
        
        social_responsibility = st.slider(
            "I care whether a clothing brand demonstrates social responsibility",
            min_value=1,
            max_value=5,
            value=3,
            help="Is brand social responsibility important to you?",
            key=f"social_resp_{st.session_state.form_key}"
        )
        
        st.markdown("**Values & Beliefs:**")
        
        muslim_brands = st.slider(
            "I prefer to buy clothing from Muslim brands or those that follow Islamic principles",
            min_value=1,
            max_value=5,
            value=3,
            help="Do you prefer brands aligned with Islamic principles?",
            key=f"muslim_{st.session_state.form_key}"
        )
        
        reuse = st.slider(
            "I try to reuse or recycle my old clothes",
            min_value=1,
            max_value=5,
            value=3,
            help="Do you actively reuse or recycle your clothing?",
            key=f"reuse_{st.session_state.form_key}"
        )
    
    st.divider()
    
    # Predict button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn1:
        reset_button = st.button(
            "🔄 Reset All",
            use_container_width=True,
            help="Clear all selections and start over"
        )
    
    with col_btn2:
        predict_button = st.button(
            "🎯 Predict My Segment",
            type="primary",
            use_container_width=True
        )
    
    # Handle reset - increment form_key to force all widgets to recreate with defaults
    if reset_button:
        st.session_state.form_key += 1
        st.session_state.show_results = False
        st.session_state.prediction_results = None
        st.rerun()
    
    # Make prediction when button is clicked
    if predict_button:
        # Prepare input data
        input_data = {
            'Gender': gender,
            'Marital Status': marital_status,
            'Age': age,
            'Occupation': occupation,
            'Frequency of purchase per month': frequency,
            'Main method of purchase': purchase_method_str,
            top5_likert[0]: recycling,
            top5_likert[1]: biodegradable,
            top5_likert[2]: muslim_brands,
            top5_likert[3]: social_responsibility,
            top5_likert[4]: reuse
        }
        
        with st.spinner("🔮 Analyzing your profile..."):
            predicted_cluster, confidence, probabilities, success, error_msg = make_prediction(input_data)
        
        # Store results in session state
        if success:
            st.session_state.show_results = True
            st.session_state.prediction_results = {
                'predicted_cluster': predicted_cluster,
                'confidence': confidence,
                'probabilities': probabilities
            }
        else:
            st.error(f"❌ Prediction failed: {error_msg}")
            st.info("Please check your inputs and try again. If the problem persists, contact support.")
    
    # Display results from session state (persists even after widget changes)
    if st.session_state.show_results and st.session_state.prediction_results:
        results = st.session_state.prediction_results
        predicted_cluster = results['predicted_cluster']
        confidence = results['confidence']
        probabilities = results['probabilities']
        
        st.success("✅ Prediction Complete!")
            
        # Display main result
        st.markdown("---")
        st.markdown("## 🎯 Your Prediction Results")
            
        # Result card
        cluster_color = cluster_info[predicted_cluster]["color"]
        st.markdown(f"""
        <div style='padding: 2rem; border-radius: 10px; background: linear-gradient(135deg, {cluster_color}22 0%, {cluster_color}44 100%); border-left: 5px solid {cluster_color};'>
            <h2 style='color: {cluster_color}; margin: 0;'>🏆 {predicted_cluster}</h2>
            <p style='font-size: 1.1rem; margin-top: 1rem;'>{cluster_info[predicted_cluster]["description"]}</p>
        </div>
        """, unsafe_allow_html=True)
            
        st.markdown("")
            
        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
            
        with metric_col1:
            st.metric(
                "Prediction Confidence",
                f"{confidence*100:.1f}%",
                help="How confident the model is in this prediction"
            )
            
        with metric_col2:
            confidence_level = "Very High" if confidence >= 0.8 else "High" if confidence >= 0.6 else "Moderate" if confidence >= 0.4 else "Low"
            st.metric(
                "Confidence Level",
                confidence_level
            )
            
        with metric_col3:
            second_highest = sorted(probabilities, reverse=True)[1]
            certainty = "Clear" if (confidence - second_highest) > 0.2 else "Borderline"
            st.metric(
                "Prediction Certainty",
                certainty
            )
            
        # Probability breakdown
        st.markdown("---")
        st.markdown("### 📊 Detailed Probability Breakdown")
            
        prob_df = pd.DataFrame({
            'Segment': [cluster_names[i] for i in range(len(probabilities))],
            'Probability': probabilities * 100
        }).sort_values('Probability', ascending=True)
            
        # Create horizontal bar chart using Plotly
        fig = go.Figure()
            
        colors = [cluster_info[segment]["color"] for segment in prob_df['Segment']]
            
        fig.add_trace(go.Bar(
            y=prob_df['Segment'],
            x=prob_df['Probability'],
            orientation='h',
            marker=dict(color=colors),
            text=[f"{p:.1f}%" for p in prob_df['Probability']],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
        ))
            
        fig.update_layout(
            title="Probability of Belonging to Each Segment",
            xaxis_title="Probability (%)",
            yaxis_title="",
            height=300,
            showlegend=False,
            xaxis=dict(range=[0, 100])
        )
            
        st.plotly_chart(fig, use_container_width=True)
            
        # Segment characteristics
        st.markdown("---")
        st.markdown(f"### 🎯 About Your Segment: {predicted_cluster}")
            
        char_col1, char_col2 = st.columns(2)
            
        with char_col1:
            st.markdown("**👥 Key Characteristics:**")
            for char in cluster_info[predicted_cluster]["characteristics"]:
                st.markdown(f"- {char}")
            
        with char_col2:
            st.markdown("**📢 Marketing Recommendations:**")
            for rec in cluster_info[predicted_cluster]["marketing"]:
                st.markdown(f"- {rec}")
            
            # What this means
            st.markdown("---")
            st.info(f"""
        **💡 What This Means:**
            
        You share characteristics with the **{predicted_cluster}** consumer segment. 
        This means you're likely to respond well to marketing strategies that align with this segment's preferences.
            
        **Confidence Interpretation:**
        - Your prediction confidence is {confidence*100:.1f}%, which is considered **{confidence_level.lower()}**.
        - {"This indicates a strong match with this segment." if confidence >= 0.7 else "You may share some characteristics with other segments as well."}
        """)

# Tab 2: Batch analysis (example like for companies usage)
with tab2:
    st.markdown("## 🏢 Batch Customer Analysis")
    st.markdown("Upload a CSV file with customer data to predict clusters for multiple customers at once")
    
    # Instructions
    with st.expander("📋 **Instructions & CSV Format Requirements**", expanded=True):
        st.markdown("""
        ### Required Columns in Your CSV:
        
        Your CSV file must contain the following columns with **exact names**:
        
        #### Demographics:
        - `Gender` - Values: Female, Male
        - `Marital Status` - Values: Single, Married, Widowed
        - `Age` - Values: 19-25, 26-35, 36-45, 45 or above
        - `Occupation` - Values: Student, Private office worker, Government servant, Freelancer, Business owner, Other
        
        #### Shopping Behavior:
        - `Frequency of purchase per month` - Values: Never, Occasionally (≤1 item), Moderate (2–3 items), Frequently (≥4 items)
        - `Main method of purchase` - Values: Physical Store, Online platform, Cross-border platform, Social Media (can be comma-separated for multiple)
        
        #### Preference Scales (1-5):
        - `I am likely to participate in recycling programs offered by clothing brands.`
        - `I prefer clothing made from biodegradable or recyclable materials.`
        - `I prefer to buy clothing from Muslim brands or those that follow Islamic principles.`
        - `I care whether a clothing brand demonstrates social responsibility.`
        - `I try to reuse or recycle my old clothes.`
        """)
    
    st.divider()
    
    # File upload
    st.markdown("### 📤 Upload Your Customer Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing customer data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            input_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(input_df)} customers")
            
            # Show preview
            with st.expander("👀 Preview Uploaded Data (First 5 Rows)", expanded=False):
                st.dataframe(input_df.head())
            
            # Validate columns
            required_cols = single_response_cols + multi_response_cols + top5_likert
            missing_cols = [col for col in required_cols if col not in input_df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                st.info("Please ensure your CSV contains all required columns. Download the template above for reference.")
            else:
                st.success("✅ All required columns found!")
                
                # Predict button
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                        with st.spinner(f"🔮 Analyzing {len(input_df)} customers..."):
                            results_df, success, error_msg = make_batch_predictions(input_df)
                        
                        if success:
                            st.success("✅ Batch prediction completed!")
                            
                            # Create summary
                            summary = create_batch_summary(results_df)
                            
                            # Display summary metrics
                            st.markdown("---")
                            st.markdown("## 📊 Analysis Summary")
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            
                            with metric_col1:
                                st.metric("Total Customers", summary['total_customers'])
                            
                            with metric_col2:
                                st.metric("Avg Confidence", f"{summary['avg_confidence']*100:.1f}%")
                            
                            with metric_col3:
                                st.metric("High Confidence", f"{summary['high_confidence_count']} ({summary['high_confidence_count']/summary['total_customers']*100:.0f}%)")
                            
                            with metric_col4:
                                st.metric("Low Confidence", f"{summary['low_confidence_count']} ({summary['low_confidence_count']/summary['total_customers']*100:.0f}%)")
                            
                            # Cluster distribution
                            st.markdown("---")
                            st.markdown("### 🎯 Customer Cluster Distribution")
                            
                            dist_col1, dist_col2 = st.columns([1, 1])
                            
                            with dist_col1:
                                # Pie chart
                                cluster_dist = pd.DataFrame.from_dict(
                                    summary['cluster_distribution'], 
                                    orient='index', 
                                    columns=['Count']
                                ).reset_index()
                                cluster_dist.columns = ['Cluster', 'Count']
                                cluster_dist['Percentage'] = (cluster_dist['Count'] / cluster_dist['Count'].sum() * 100).round(1)
                                
                                fig_pie = px.pie(
                                    cluster_dist,
                                    values='Count',
                                    names='Cluster',
                                    title='Cluster Distribution',
                                    color='Cluster',
                                    color_discrete_map={
                                        'Mainstream Casual': '#4CAF50',
                                        'Fashion-Forward Modest Consumer': '#E91E63',
                                        'Traditional Functionalist': '#2196F3'
                                    }
                                )
                                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with dist_col2:
                                # Bar chart
                                fig_bar = go.Figure(data=[
                                    go.Bar(
                                        x=cluster_dist['Cluster'],
                                        y=cluster_dist['Count'],
                                        text=cluster_dist['Count'],
                                        textposition='auto',
                                        marker_color=[cluster_info[seg]['color'] for seg in cluster_dist['Cluster']]
                                    )
                                ])
                                fig_bar.update_layout(
                                    title='Customer Count by Cluster',
                                    xaxis_title='Cluster',
                                    yaxis_title='Number of Customers',
                                    showlegend=False
                                )
                                st.plotly_chart(fig_bar, use_container_width=True)
                                
                                # Display table
                                st.markdown("**Distribution Table:**")
                                st.dataframe(cluster_dist, hide_index=True, use_container_width=True)
                            
                            # Confidence distribution
                            st.markdown("---")
                            st.markdown("### 📈 Prediction Confidence Distribution")
                            
                            fig_conf = px.histogram(
                                results_df,
                                x='Prediction_Confidence',
                                nbins=20,
                                title='Distribution of Prediction Confidence',
                                labels={'Prediction_Confidence': 'Confidence Score', 'count': 'Number of Customers'},
                                color_discrete_sequence=['#0078D4']
                            )
                            fig_conf.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="High Confidence Threshold")
                            fig_conf.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Low Confidence Threshold")
                            st.plotly_chart(fig_conf, use_container_width=True)
                            
                            # Detailed results table
                            st.markdown("---")
                            st.markdown("### 📋 Detailed Prediction Results")
                            
                            # Show key columns
                            display_cols = [col for col in results_df.columns if col not in feature_columns]
                            
                            # Add confidence level
                            results_df['Confidence_Level'] = results_df['Prediction_Confidence'].apply(
                                lambda x: 'Very High' if x >= 0.8 else 'High' if x >= 0.6 else 'Moderate' if x >= 0.4 else 'Low'
                            )
                            
                            st.dataframe(
                                results_df[['Predicted_Cluster_Name', 'Prediction_Confidence', 'Confidence_Level'] + 
                                          [col for col in results_df.columns if col.startswith('Prob_')]],
                                use_container_width=True,
                                height=400
                            )
                            
                            # Download options
                            st.markdown("---")
                            st.markdown("### 💾 Download Results")
                            
                            download_col1, download_col2 = st.columns(2)
                            
                            with download_col1:
                                # CSV download
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name="customer_cluster_predictions.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with download_col2:
                                # Excel download
                                excel_buffer = convert_df_to_excel(results_df)
                                st.download_button(
                                    label="📥 Download Results as Excel",
                                    data=excel_buffer,
                                    file_name="customer_cluster_predictions.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            # Cluster-specific insights
                            st.markdown("---")
                            st.markdown("### 💡 Actionable Insights by Cluster")
                            
                            for cluster_name, count in summary['cluster_distribution'].items():
                                percentage = (count / summary['total_customers']) * 100
                                cluster_customers = results_df[results_df['Predicted_Cluster_Name'] == cluster_name]
                                avg_conf = cluster_customers['Prediction_Confidence'].mean()
                                
                                with st.expander(f"**{cluster_name}** - {count} customers ({percentage:.1f}%)", expanded=False):
                                    st.markdown(f"**Average Confidence:** {avg_conf*100:.1f}%")
                                    st.markdown(f"**Description:** {cluster_info[cluster_name]['description']}")
                                    
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.markdown("**Key Characteristics:**")
                                        for char in cluster_info[cluster_name]["characteristics"]:
                                            st.markdown(f"- {char}")
                                    
                                    with col_b:
                                        st.markdown("**Recommended Actions:**")
                                        for rec in cluster_info[cluster_name]["marketing"]:
                                            st.markdown(f"- {rec}")
                        
                        else:
                            st.error(f"❌ Batch prediction failed: {error_msg}")
                            st.info("Please check your CSV format and try again.")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format.")

# TAB 3: ABOUT CLUSTERS (Original)
with tab3:
    st.markdown("## 📊 Understanding the Three Consumer Clusters")
    st.markdown("Our analysis identified three distinct consumer clusters in the Indonesian textile market:")

    for i, (cluster_name, info) in enumerate(cluster_info.items(), 1):
        with st.expander(f"**{i}. {cluster_name}**", expanded=(i==1)):
            st.markdown(f"**Description:** {info['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Key Characteristics:**")
                for char in info["characteristics"]:
                    st.markdown(f"- {char}")
            
            with col2:
                st.markdown("**Marketing Approach:**")
                for rec in info["marketing"]:
                    st.markdown(f"- {rec}")

# TAB 4: HOW IT WORKS (Updated)
with tab4:
    st.markdown("## ℹ️ How This Prediction Tool Works")
    
    st.markdown("""
    ### 🤖 The Model
    
    This prediction tool uses a **Logistic Regression** machine learning model trained on data from 
    **720 Indonesian textile consumers**. The model analyzes demographics, shopping behavior, 
    and value preferences to predict which consumer cluster best aligns with each customer.
    
    ### 📊 Model Performance
    """)
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    with perf_col1:
        st.metric("Overall Accuracy", "~84%")
    with perf_col2:
        st.metric("Training Samples", "720")
    with perf_col3:
        st.metric("Consumer Clusters", "3")
    
    st.markdown("""
    ### 🔍 What the Model Analyzes
    
    The model considers multiple factors to make its prediction:
    
    1. **Demographics** (35% importance)
       - Gender, Age, Marital Status, Occupation
       
    2. **Shopping Behavior** (30% importance)
       - Purchase frequency, Preferred channels
       
    3. **Values & Beliefs** (35% importance)
       - Sustainability attitudes, Religious preferences, Social responsibility
    
    ### 🎯 Two Usage Modes
    
    **1. Individual Prediction (Tab 1):**
    - Perfect for consumers wanting to discover their cluster
    - Interactive input form for all attributes
    - Detailed visualization of results
    - Personalized insights
    
    **2. Batch Analysis (Tab 2):**
    - Designed for companies with customer databases
    - Upload CSV with multiple customer records
    - Get predictions for hundreds/thousands at once
    - Download results for CRM integration
    - Aggregate insights and cluster distribution
    
    ### 🏢 Business Applications
    
    Companies can use batch predictions to:
    - **Cluster existing customers** for targeted marketing
    - **Personalize product recommendations** based on predicted clusters
    - **Optimize inventory** by understanding cluster distribution
    - **Design targeted campaigns** for each cluster
    - **Improve customer retention** with cluster-specific strategies

    ### ⚠️ Important Notes
    
    - Predictions are based on statistical patterns and may not capture individual nuances
    - Use results as guidance, not absolute truth
    - The model performs best for profiles similar to the training data (Indonesian consumers)
    - Confidence scores indicate prediction reliability
    - For batch analysis, ensure data quality for best results
    """)

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎯 Quick Navigation")
    st.markdown("""
    - **Individual Prediction**: Single customer analysis
    - **Batch Analysis**: Multiple customers (CSV upload)
    - **About Clusters**: Learn about consumer types
    - **How It Works**: Model methodology
    """)
    
    st.divider()
    
    st.markdown("## 📊 Model Information")
    st.markdown(f"""
    - **Model Type**: Logistic Regression
    - **Accuracy**: 84%
    - **Training Data**: 720 consumers
    - **Clusters**: {len(cluster_names)}
    - **Features**: {len(feature_columns)} （encoded, 11 original features）
    """)
    
    st.divider()

    st.markdown("## 🔗 Related Resources")
    st.markdown("""
    - [View Full Dashboard](#) *(https://365umedumy-my.sharepoint.com/:u:/g/personal/23083896_siswa365_um_edu_my/IQChCOw4XqXvQrHzSB3YuY2gAf4C30J2aNsnjewoRgStBIo?e=1hyN4C)*
    """)
    
    st.divider()
    
    st.caption("Developed by 23083896 for WQD7025 Data Science Research Project")