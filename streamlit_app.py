import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import google.generativeai as genai
from io import StringIO
import hashlib
import socket
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io
import os

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyBCo7F5JUROk1EgF-mMefQAKz3JPR6VzaU"  # Replace with your actual Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Function to calculate data quality score
def calculate_data_quality_score(df, issues):
    score = 100
    total_rows = df.shape[0]
    total_cols = df.shape[1]
    
    # Deduct points for missing values
    total_missing = sum(issues["Missing Values"].values())
    missing_penalty = (total_missing / (total_rows * total_cols)) * 30  # Max 30 points
    score -= missing_penalty
    
    # Deduct points for duplicates
    duplicates_penalty = (issues["Duplicates"] / total_rows) * 20  # Max 20 points
    score -= duplicates_penalty
    
    # Deduct points for outliers
    total_outliers = sum(issues["Outliers"].values())
    outliers_penalty = (total_outliers / total_rows) * 20  # Max 20 points
    score -= outliers_penalty
    
    # Deduct points for data type issues
    type_issues_penalty = len(issues["Data Type Issues"]) * 5  # 5 points per issue
    score -= type_issues_penalty
    
    return max(0, round(score))

# Function to analyze data issues
def analyze_data(df):
    issues = {
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicates": df.duplicated().sum(),
        "Outliers": {col: ((df[col] - df[col].mean()).abs() > 3 * df[col].std()).sum() 
                     for col in df.select_dtypes(include=[np.number]).columns},
        "Data Type Issues": {col: "Potential mismatch" for col in df.columns 
                            if df[col].dtype == 'object' and df[col].str.match(r'^-?\d*\.?\d+$').all()}
    }
    return issues

# Function to clean data
def clean_data(df, options, custom_rules=None):
    cleaned_df = df.copy()
    
    for col, method in options["missing_values"].items():
        if method == "mean" and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
        elif method == "median" and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
        elif method == "mode" and col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else np.nan)
        elif method == "drop":
            cleaned_df = cleaned_df.dropna(subset=[col])

    if options["remove_duplicates"]:
        cleaned_df = cleaned_df.drop_duplicates()

    for col, threshold in options["outliers"].items():
        if col in cleaned_df.columns and threshold > 0:
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            cleaned_df = cleaned_df[z_scores < threshold]

    for col in options["fix_types"]:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')

    if custom_rules:
        for col, (old_val, new_val) in custom_rules.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].replace(old_val, new_val)

    return cleaned_df

# Gemini API Functions with Language Support
def summarize_data(df, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        summary_prompt = f"{lang_instruction}\nSummarize the following dataset:\n{df.describe().to_string()}\nProvide a concise summary of key trends and insights."
        response = model.generate_content(summary_prompt)
        return response.text
    except Exception as e:
        return f"Error summarizing data with Gemini API: {e}"

def detect_anomalies(df, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        anomaly_prompt = f"{lang_instruction}\nAnalyze this dataset for potential anomalies that might indicate cybersecurity issues (e.g., unusual patterns, data tampering):\n{df.head(10).to_string()}\nHighlight any suspicious patterns."
        response = model.generate_content(anomaly_prompt)
        return response.text
    except Exception as e:
        return f"Error detecting anomalies with Gemini API: {e}"

def generate_cleaning_report(original_df, cleaned_df, issues, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        report_prompt = f"{lang_instruction}\nGenerate a detailed report of the data cleaning process:\nOriginal Issues: {issues}\nOriginal Rows: {original_df.shape[0]}\nCleaned Rows: {cleaned_df.shape[0]}\nExplain the cleaning actions taken and their impact."
        response = model.generate_content(report_prompt)
        return response.text
    except Exception as e:
        return f"Error generating report with Gemini API: {e}"

def answer_query(df, query, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        query_prompt = f"{lang_instruction}\nAnalyze this dataset:\n{df.head(10).to_string()}\nUser question: {query}"
        response = model.generate_content(query_prompt)
        return response.text
    except Exception as e:
        return f"Error answering query with Gemini API: {e}"

def suggest_cleaning_actions(df, issues, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        suggestion_prompt = f"{lang_instruction}\nGiven the following dataset issues:\n{issues}\nSuggest specific cleaning actions to improve data quality (e.g., handle missing values, remove outliers)."
        response = model.generate_content(suggestion_prompt)
        return response.text
    except Exception as e:
        return f"Error suggesting cleaning actions with Gemini API: {e}"

def detect_sensitive_data(df, language="English"):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        lang_instruction = f"Provide the response in {language}."
        sensitive_prompt = f"{lang_instruction}\nAnalyze this dataset for potentially sensitive data (e.g., names, emails, phone numbers):\n{df.head(10).to_string()}\nSuggest columns to anonymize."
        response = model.generate_content(sensitive_prompt)
        return response.text
    except Exception as e:
        return f"Error detecting sensitive data with Gemini API: {e}"

# Function to compute dataset hash
def compute_dataset_hash(df):
    df_string = df.to_string()
    return hashlib.sha256(df_string.encode()).hexdigest()

# Function to get user IP (for audit trail)
def get_user_ip():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return "Unknown IP"

# Function to generate PDF report
def generate_pdf_report(cleaned_df, summary, anomalies, report):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(1 * inch, height - 1 * inch, "Data Analytics Tools")

    c.setFont("Helvetica", 12)
    y_position = height - 2 * inch
    c.drawString(1 * inch, y_position, "Dataset Preview:")
    y_position -= 0.5 * inch
    for i, row in cleaned_df.head(5).iterrows():
        c.drawString(1.5 * inch, y_position, str(row.to_dict()))
        y_position -= 0.3 * inch

    y_position -= 0.5 * inch
    c.drawString(1 * inch, y_position, "Summary:")
    y_position -= 0.3 * inch
    for line in summary.split("\n")[:5]:
        c.drawString(1.5 * inch, y_position, line[:80])
        y_position -= 0.3 * inch

    y_position -= 0.5 * inch
    c.drawString(1 * inch, y_position, "Cybersecurity Insights:")
    y_position -= 0.3 * inch
    for line in anomalies.split("\n")[:5]:
        c.drawString(1.5 * inch, y_position, line[:80])
        y_position -= 0.3 * inch

    y_position -= 0.5 * inch
    c.drawString(1 * inch, y_position, "Cleaning Report:")
    y_position -= 0.3 * inch
    for line in report.split("\n")[:5]:
        c.drawString(1.5 * inch, y_position, line[:80])
        y_position -= 0.3 * inch

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit App
st.set_page_config(page_title="Data Analytics Tools", layout="wide")

# Theme Options
themes = {
    "Dark Cybersecurity": """
        .main {background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%); color: #e0e0e0;}
        .sidebar .sidebar-content {background: #1b263b; border-radius: 12px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);}
        .stTabs {background: #1b263b; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);}
        .stButton>button {background: #00d4ff; color: #1a1a2e;}
        .stButton>button:hover {background: #00b4d8; color: #ffffff;}
        .stDownloadButton>button {background: #ff4d4d; color: #ffffff;}
        h1, h2, h3, h4, h5, h6 {color: #00d4ff !important;}
        p, div, span, label {color: #e0e0e0 !important;}
        .stDataFrame {background: #1b263b;}
        .stDataFrame table {color: #e0e0e0 !important;}
        .stSuccess {background: #1a3c34 !important; color: #00ff00 !important;}
        .stWarning {background: #4a1a1a !important; color: #ff4d4d !important;}
    """,
    "Light Minimal": """
        .main {background: #f5f5f5; color: #333333;}
        .sidebar .sidebar-content {background: #ffffff; border-radius: 12px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
        .stTabs {background: #ffffff; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
        .stButton>button {background: #4CAF50; color: #ffffff;}
        .stButton>button:hover {background: #45a049; color: #ffffff;}
        .stDownloadButton>button {background: #ff4d4d; color: #ffffff;}
        h1, h2, h3, h4, h5, h6 {color: #4CAF50 !important;}
        p, div, span, label {color: #333333 !important;}
        .stDataFrame {background: #ffffff;}
        .stDataFrame table {color: #333333 !important;}
        .stSuccess {background: #e6f4e6 !important; color: #2e7d32 !important;}
        .stWarning {background: #ffebee !important; color: #d32f2f !important;}
    """,
    "Neon Tech": """
        .main {background: linear-gradient(135deg, #1a0033 0%, #330066 100%); color: #e0e0e0;}
        .sidebar .sidebar-content {background: #1a0033; border-radius: 12px; padding: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);}
        .stTabs {background: #1a0033; padding: 15px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.5);}
        .stButton>button {background: #ff00ff; color: #ffffff;}
        .stButton>button:hover {background: #cc00cc; color: #ffffff;}
        .stDownloadButton>button {background: #ff4d4d; color: #ffffff;}
        h1, h2, h3, h4, h5, h6 {color: #ff00ff !important;}
        p, div, span, label {color: #e0e0e0 !important;}
        .stDataFrame {background: #1a0033;}
        .stDataFrame table {color: #e0e0e0 !important;}
        .stSuccess {background: #1a3c34 !important; color: #00ff00 !important;}
        .stWarning {background: #4a1a1a !important; color: #ff4d4d !important;}
    """
}

# Theme and Language Selection
col1, col2 = st.columns([1, 1])
with col1:
    theme = st.selectbox("Select Theme / थीम चुनें", list(themes.keys()))
with col2:
    language = st.selectbox("Select Language / भाषा चुनें", ["English", "Hindi"])

# Apply selected theme
st.markdown(f"""
    <style>
    {themes[theme]}
    div[data-baseweb="tab"] > div > div {{color: inherit !important; font-weight: bold;}}
    .stButton>button {{border-radius: 8px; padding: 8px 16px; font-weight: bold; transition: all 0.3s ease;}}
    .stDownloadButton>button {{border-radius: 8px;}}
    .stSelectbox, .stSlider, .stTextInput {{background: #16213e; border-radius: 8px; padding: 5px; color: #e0e0e0 !important;}}
    .stTextInput>div>input {{color: #e0e0e0 !important;}}
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.title("🔒 CyberClean Data Tool")
st.markdown("Protect, clean, and analyze your data with advanced AI-powered insights in English and Hindi!")

# Sidebar for upload and settings
with st.sidebar:
    st.header("Upload & Configure / अपलोड और कॉन्फ़िगर करें")
    uploaded_files = st.file_uploader("Upload CSV(s) / CSV अपलोड करें", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        st.subheader("Global Cleaning Options / वैश्विक सफाई विकल्प")
        skip_bad_lines = st.checkbox("Skip rows with parsing errors / पार्सिंग त्रुटियों वाली पंक्तियों को छोड़ें", value=True)
        remove_duplicates = st.checkbox("Remove Duplicates / डुप्लिकेट हटाएं", value=True)

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# Process multiple files
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Processing File: {uploaded_file.name}")
        try:
            df = pd.read_csv(uploaded_file, on_bad_lines='skip' if skip_bad_lines else 'error')
            original_df = df.copy()
            st.session_state.history.append({"df": original_df, "timestamp": datetime.now(), "file_name": uploaded_file.name})

            issues = analyze_data(df)
            quality_score = calculate_data_quality_score(df, issues)

            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "📊 Analysis / विश्लेषण", 
                "⚙️ Cleaning Options / सफाई विकल्प", 
                "📋 Cleaned Data / साफ किया गया डेटा", 
                "🔍 Visualizations / दृश्य", 
                "🛡️ Cybersecurity Insights / साइबर सुरक्षा अंतर्दृष्टि",
                "📜 Version History / संस्करण इतिहास"
            ])

            # Tab 1: Analysis
            with tab1:
                st.subheader("Data Analysis / डेटा विश्लेषण")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Missing Values per Column / प्रति कॉलम लुप्त मान**")
                    st.json(issues["Missing Values"])
                    st.metric("Total Duplicates / कुल डुप्लिकेट", issues["Duplicates"])
                with col2:
                    st.write("**Outliers per Numeric Column / संख्यात्मक कॉलम प्रति आउटलायर्स**")
                    st.json(issues["Outliers"])
                    if issues["Data Type Issues"]:
                        st.write("**Potential Data Type Issues / संभावित डेटा प्रकार समस्याएं**")
                        st.json(issues["Data Type Issues"])

                st.subheader("Data Quality Score / डेटा गुणवत्ता स्कोर")
                st.metric("Quality Score", quality_score)
                st.write("**Suggestions to Improve Quality / गुणवत्ता सुधार के सुझाव**")
                suggestions = suggest_cleaning_actions(df, issues, language)
                st.write(suggestions)

                # Gemini-powered summary
                st.subheader("Dataset Summary (Powered by Gemini) / डेटासेट सारांश (जेमिनी द्वारा संचालित)")
                summary = summarize_data(df, language)
                st.write(summary)

            # Tab 2: Cleaning Options
            with tab2:
                st.subheader("Configure Cleaning / सफाई कॉन्फ़िगर करें")
                
                st.write("**Handle Missing Values / लुप्त मानों को संभालें**")
                missing_options = {}
                for col in df.columns:
                    if issues["Missing Values"][col] > 0:
                        method = st.selectbox(f"{col}", ["mean", "median", "mode", "drop", "ignore"], key=f"miss_{col}_{uploaded_file.name}", help=f"Choose how to handle missing values in {col}.")
                        if method != "ignore":
                            missing_options[col] = method

                st.write("**Handle Outliers / आउटलायर्स को संभालें**")
                outlier_options = {}
                for col in issues["Outliers"].keys():
                    threshold = st.slider(f"{col} (Z-score)", 0.0, 5.0, 0.0, 0.1, key=f"out_{col}_{uploaded_file.name}", help=f"Set Z-score threshold for outlier removal in {col}.")
                    if threshold > 0:
                        outlier_options[col] = threshold

                st.write("**Fix Data Types / डेटा प्रकार ठीक करें**")
                type_options = st.multiselect("Convert to numeric / संख्यात्मक में परिवर्तित करें", list(issues["Data Type Issues"].keys()), key=f"type_{uploaded_file.name}")

                st.write("**Custom Rules / कस्टम नियम**")
                with st.expander("Add Replacement Rules / प्रतिस्थापन नियम जोड़ें"):
                    custom_rules = {}
                    rule_col = st.selectbox("Column / कॉलम", df.columns, key=f"rule_col_{uploaded_file.name}")
                    old_val = st.text_input("Value to Replace / प्रतिस्थापित करने वाला मान", key=f"old_val_{uploaded_file.name}")
                    new_val = st.text_input("Replace With / इसके साथ प्रतिस्थापित करें", key=f"new_val_{uploaded_file.name}")
                    if st.button("Add Rule / नियम जोड़ें", key=f"add_rule_{uploaded_file.name}"):
                        custom_rules[rule_col] = (old_val, new_val)
                        st.success(f"Added rule: {rule_col}: {old_val} -> {new_val}")

                if st.button("Apply Cleaning / सफाई लागू करें", key=f"apply_{uploaded_file.name}"):
                    options = {
                        "missing_values": missing_options,
                        "remove_duplicates": remove_duplicates,
                        "outliers": outlier_options,
                        "fix_types": type_options
                    }
                    cleaned_df = clean_data(df, options, custom_rules)
                    st.session_state.history.append({"df": cleaned_df, "timestamp": datetime.now(), "file_name": uploaded_file.name})
                    st.session_state.cleaned_df = cleaned_df

            # Tab 3: Cleaned Data
            with tab3:
                if "cleaned_df" in st.session_state:
                    st.subheader("Cleaned Data Preview / साफ किया गया डेटा पूर्वावलोकन")
                    st.write(f"Rows: {st.session_state.cleaned_df.shape[0]}, Columns: {st.session_state.cleaned_df.shape[1]}")
                    st.data_editor(st.session_state.cleaned_df, use_container_width=True)

                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        csv = st.session_state.cleaned_df.to_csv(index=False)
                        st.download_button("Download Cleaned Data / साफ किया गया डेटा डाउनलोड करें", csv, "cleaned_data.csv", "text/csv")
                    with col2:
                        if len(st.session_state.history) > 1 and st.button("Undo Last Cleaning / अंतिम सफाई पूर्ववत करें"):
                            st.session_state.history.pop()
                            st.session_state.cleaned_df = st.session_state.history[-1]
                            st.success("Reverted to previous state! / पिछले स्थिति में लौटाया गया!")

                    # Generate cleaning report
                    st.subheader("Cleaning Report (Powered by Gemini) / सफाई रिपोर्ट (जेमिनी द्वारा संचालित)")
                    report = generate_cleaning_report(original_df, st.session_state.cleaned_df, issues, language)
                    st.write(report)
                    with col3:
                        st.download_button("Download Cleaning Report / सफाई रिपोर्ट डाउनलोड करें", report, "cleaning_report.txt", "text/plain")

                    # Generate PDF report
                    st.subheader("Export Visual Report / दृश्य रिपोर्ट निर्यात करें")
                    anomalies = detect_anomalies(df, language)
                    pdf_buffer = generate_pdf_report(st.session_state.cleaned_df, summary, anomalies, report)
                    st.download_button("Download PDF Report / PDF रिपोर्ट डाउनलोड करें", pdf_buffer, "cyberclean_report.pdf", "application/pdf")

            # Tab 4: Visualizations
            with tab4:
                if "cleaned_df" in st.session_state:
                    st.subheader("Data Visualizations / डेटा दृश्य")
                    plot_col = st.selectbox("Select Column to Visualize / दृश्य के लिए कॉलम चुनें", st.session_state.cleaned_df.columns, key=f"plot_col_{uploaded_file.name}")
                    plot_type = st.selectbox("Plot Type / प्लॉट प्रकार", ["Histogram", "Box Plot", "Line Chart"], key=f"plot_type_{uploaded_file.name}")
                    
                    if plot_type == "Histogram":
                        fig = px.histogram(st.session_state.cleaned_df, x=plot_col)
                    elif plot_type == "Box Plot":
                        fig = px.box(st.session_state.cleaned_df, y=plot_col)
                    else:
                        fig = px.line(st.session_state.cleaned_df, y=plot_col)
                    st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Cybersecurity Insights
            with tab5:
                st.subheader("Cybersecurity Insights (Powered by Gemini) / साइबर सुरक्षा अंतर्दृष्टि (जेमिनी द्वारा संचालित)")
                
                # Data Integrity Hash
                st.write("**Data Integrity Hash (SHA-256) / डेटा अखंडता हैश (SHA-256):**")
                data_hash = compute_dataset_hash(df)
                st.write(data_hash)

                # Audit Trail
                st.write("**Audit Trail / ऑडिट ट्रेल:**")
                user_ip = get_user_ip()
                st.write(f"Processed by IP: {user_ip} at {datetime.now()}")

                # Sensitive Data Detection
                st.write("**Sensitive Data Detection / संवेदनशील डेटा पहचान:**")
                sensitive_data = detect_sensitive_data(df, language)
                st.write(sensitive_data)

                # Anomalies
                st.write("**Potential Anomalies (Cybersecurity Scan) / संभावित विसंगतियाँ (साइबर सुरक्षा स्कैन):**")
                anomalies = detect_anomalies(df, language)
                st.write(anomalies)

                # Chat interface for data queries
                st.subheader("Ask About Your Data / अपने डेटा के बारे में पूछें")
                user_query = st.text_input("Enter your question (e.g., 'What are the main trends in my dataset?') / अपना प्रश्न दर्ज करें (उदाहरण: 'मेरे डेटासेट में मुख्य रुझान क्या हैं?')", key=f"query_{uploaded_file.name}")
                if user_query:
                    response = answer_query(df, user_query, language)
                    st.write("**Answer / उत्तर:**", response)

            # Tab 6: Version History
            with tab6:
                st.subheader("Version History / संस्करण इतिहास")
                if st.session_state.history:
                    for i, version in enumerate(st.session_state.history):
                        st.write(f"Version {i+1}: {version['file_name']} at {version['timestamp']}")
                        if st.button(f"Revert to Version {i+1}", key=f"revert_{i}_{uploaded_file.name}"):
                            st.session_state.cleaned_df = version["df"]
                            st.success(f"Reverted to version {i+1}! / संस्करण {i+1} पर लौटाया गया!")

                        if i > 0:
                            if st.button(f"Compare with Previous Version", key=f"compare_{i}_{uploaded_file.name}"):
                                st.write("**Comparison / तुलना:**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"Version {i}:")
                                    st.dataframe(st.session_state.history[i-1]["df"].head())
                                with col2:
                                    st.write(f"Version {i+1}:")
                                    st.dataframe(version["df"].head())

        except Exception as e:
            st.error(f"Error / त्रुटि: {e}")
else:
    st.info("Upload a CSV file to begin! / शुरू करने के लिए एक CSV फ़ाइल अपलोड करें!")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Built with RoyDev❤️ using Streamlit and Gemini API by xAI")
