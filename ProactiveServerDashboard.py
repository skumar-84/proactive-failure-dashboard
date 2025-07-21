import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
import requests
import base64

# --- User Role Simulation ---
user_role = st.sidebar.selectbox("Select User Role", ["Viewer", "Operator", "Admin"])

# --- Display Logo ---
st.image("Image_1.jpg", width=720)
st.title("Proactive Server/Device Failure Detection Dashboard")

# --- Load Event Logs ---
@st.cache_data
def load_event_logs():
    xl = pd.read_excel("Hardware_Eventlog.xlsx", sheet_name=None, engine='openpyxl')
    df = pd.concat(xl.values(), keys=xl.keys()).reset_index(level=0).rename(columns={'level_0': 'Server'})
    df['Last Update'] = pd.to_datetime(df['Last Update'], errors='coerce')
    return df

event_df = load_event_logs()

# --- Generate Simulated Metric Data ---
@st.cache_data
def generate_metrics():
    servers = event_df['Server'].unique()[:10]
    timestamps = pd.date_range(end=datetime.now(), periods=72, freq='H')
    data = []
    for server in servers:
        for ts in timestamps:
            cpu = np.random.normal(50, 10)
            mem = np.random.normal(60, 15)
            disk = np.random.normal(100, 20)
            net = np.random.normal(200, 50)
            data.append([server, ts, cpu, mem, disk, net])
    df = pd.DataFrame(data, columns=['Server', 'Timestamp', 'CPU', 'Memory', 'Disk_IO', 'Network'])
    return df

metric_df = generate_metrics()

# --- Custom Thresholds ---
st.sidebar.header("Custom Alert Thresholds")
custom_thresholds = {}
for metric in ['CPU', 'Memory', 'Disk_IO', 'Network']:
    custom_thresholds[metric] = st.sidebar.slider(f"{metric} Threshold", 0, 300, int(metric_df[metric].mean() + 2 * metric_df[metric].std()))

# --- Baseline and Anomaly Detection ---
def detect_anomalies(df, custom_thresholds=None):
    anomalies = []
    for server in df['Server'].unique():
        sub = df[df['Server'] == server]
        for metric in ['CPU', 'Memory', 'Disk_IO', 'Network']:
            mean = sub[metric].mean()
            std = sub[metric].std()
            threshold = custom_thresholds.get(metric, mean + 2 * std) if custom_thresholds else mean + 2 * std
            outliers = sub[sub[metric] > threshold]
            for _, row in outliers.iterrows():
                anomalies.append({
                    'Server': server,
                    'Timestamp': row['Timestamp'],
                    'Metric': metric,
                    'Value': row[metric],
                    'Deviation': row[metric] - mean,
                    'Threshold': threshold
                })
    return pd.DataFrame(anomalies)

anomaly_df = detect_anomalies(metric_df, custom_thresholds)

# --- TTF Prediction (Simulated) ---
def predict_ttf(df):
    ttf_records = []
    for server in df['Server'].unique():
        sub = df[df['Server'] == server]
        for metric in ['CPU', 'Memory', 'Disk_IO', 'Network']:
            recent = sub.sort_values('Timestamp', ascending=False).iloc[0]
            ttf = np.random.randint(6, 48)
            ttf_records.append({
                'Server': server,
                'Metric': metric,
                'Current Value': recent[metric],
                'Predicted TTF (hrs)': ttf
            })
    return pd.DataFrame(ttf_records)

ttf_df = predict_ttf(metric_df)

# --- Cross-Device Anomaly Analysis ---
cross_anomalies = anomaly_df.groupby(['Timestamp', 'Metric']).size().reset_index(name='Count')
cross_anomalies = cross_anomalies[cross_anomalies['Count'] > 1]

# --- UI: Server Selection ---
servers = metric_df['Server'].unique()
selected_servers = st.multiselect("Select Servers for Comparison", servers, default=servers[:2])

# --- Metric Trends ---
st.subheader("Metric Trends")
for server in selected_servers:
    sub = metric_df[metric_df['Server'] == server]
    fig = px.line(sub, x='Timestamp', y=['CPU', 'Memory', 'Disk_IO', 'Network'], title=f"Metrics for {server}")
    st.plotly_chart(fig)

# --- Multi-Server Comparison ---
st.subheader("Multi-Server Metric Comparison")
metric_to_compare = st.selectbox("Select Metric for Comparison", ['CPU', 'Memory', 'Disk_IO', 'Network'])
fig = px.line(metric_df[metric_df['Server'].isin(selected_servers)], x='Timestamp', y=metric_to_compare, color='Server', title=f"{metric_to_compare} Comparison")
st.plotly_chart(fig)

# --- Anomaly Table ---
st.subheader("Detected Anomalies")
st.dataframe(anomaly_df[anomaly_df['Server'].isin(selected_servers)].sort_values('Timestamp', ascending=False))

# --- TTF Table ---
st.subheader("Time-to-Failure Predictions")
st.dataframe(ttf_df[ttf_df['Server'].isin(selected_servers)])

# --- Cross-Device Anomalies ---
st.subheader("Cross-Device Anomalies")
if not cross_anomalies.empty:
    st.dataframe(cross_anomalies.sort_values('Timestamp', ascending=False))
else:
    st.info("No cross-device anomalies detected.")

# --- Heatmap ---
st.subheader("Anomaly Heatmap")
heatmap_data = anomaly_df.pivot_table(index='Server', columns='Metric', values='Deviation', aggfunc='mean').fillna(0)
fig, ax = plt.subplots()
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# --- Trendlines/Forecasts ---
st.subheader("Trendlines & Forecasts (Simulated)")
for server in selected_servers:
    sub = metric_df[metric_df['Server'] == server]
    fig = px.line(sub, x='Timestamp', y=['CPU', 'Memory', 'Disk_IO', 'Network'], title=f"Trendlines for {server}")
    st.plotly_chart(fig)

# --- Alerting ---
st.subheader("Send Alert for Latest Anomaly")
if st.button("Send Alert"):
    if not anomaly_df.empty:
        latest = anomaly_df.sort_values('Timestamp', ascending=False).iloc[0]
        subject = f"Proactive Alert: {latest['Server']} {latest['Metric']} anomaly"
        body = f"""
        Server: {latest['Server']}
        Metric: {latest['Metric']}
        Value: {latest['Value']:.2f}
        Deviation: {latest['Deviation']:.2f}
        Threshold: {latest['Threshold']:.2f}
        Timestamp: {latest['Timestamp']}
        """
        to_emails = ['saurabh.kumar@lumen.com','ankur.kumar@lumen.com','reddy.sanath@lumen.com']  # <-- CHANGE THIS
        teams_webhook_url = "https://centurylink.webhook.office.com/webhookb2/4d2d03a2-bbe3-4cc6-9ca0-bd5fcdd4ed25@72b17115-9915-42c0-9f1b-4f98e5a4bcd2/IncomingWebhook/27117bdf354541dfa1f0cabb3731b9be/bd6e7f52-ef3e-4df8-8277-6fddc8824592/V2GTo_AMRnkYNQGtqwxxS0dq-w9nrQEHN7lpiOkbIgPPk1"  # <-- CHANGE THIS

        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = to_emails[0]
            msg['To'] = ', '.join(to_emails)
            with smtplib.SMTP('mailgate.qintra.com', 22) as server:
                server.starttls()
                server.login('saurabh.kumar@lumen.com', 'Shankar@840609')
                server.sendmail(to_emails[0], to_emails, msg.as_string())
            st.success("Email alert sent!")
        except Exception as e:
            st.error(f"Email failed: {e}")

        try:
            response = requests.post(teams_webhook_url, json={"title": subject, "text": body})
            if response.status_code == 200:
                st.success("Teams alert sent!")
            else:
                st.error("Teams alert failed.")
        except Exception as e:
            st.error(f"Teams alert failed: {e}")
    else:
        st.warning("No anomaly found to send alert.")

# --- Alert History and Export ---
st.subheader("Alert History & Export")
csv = anomaly_df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f'<a href="data:file/csv;base64,{b64}" download="anomalies.csv">Download Anomaly Data as CSV</a>'
st.markdown(href, unsafe_allow_html=True)

# --- Root Cause Suggestion (Simulated) ---
st.subheader("Root Cause Suggestions & Remediation")
if not anomaly_df.empty:
    sample = anomaly_df.iloc[0]
    st.write(f"Server: {sample['Server']}, Metric: {sample['Metric']}")
    st.write("Suggested Root Cause: Possible workload spike or hardware degradation.")
    st.write("Remediation: Check running processes, verify cooling and power supply, consider reboot or maintenance.")
