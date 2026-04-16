"""🎯 CUSTOMER SEGMENTATION - RFM ANALYSIS DASHBOARD"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(page_title="Customer Analytics", layout="wide")

C1, C2, C3, C4 = "#1a237e", "#283593", "#ff6b35", "#00d9ff"

st.markdown(f"""<style>
.header {{background: linear-gradient(135deg, {C1} 0%, {C2} 100%); padding: 40px; border-radius: 15px; color: white; margin-bottom: 30px;}}
.metric {{background: linear-gradient(135deg, #3949ab 0%, {C1} 100%); padding: 25px; border-radius: 12px; color: white; text-align: center;}}
</style>""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/customer_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df

df = load_data()
st.markdown(f'<div class="header"><h1>👥 Customer Segmentation & RFM</h1><p>Advanced Behavioral Analysis</p></div>', unsafe_allow_html=True)

# RFM Calculation
rfm = df.groupby('customer_id').agg({
    'date': lambda x: (df['date'].max() - x.max()).days,
    'order_id': 'count',
    'order_value': 'sum'
}).rename(columns={'date': 'recency', 'order_id': 'frequency', 'order_value': 'monetary'})

rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 5, labels=[1,2,3,4,5], duplicates='drop')
rfm['rfm_score'] = rfm['r_score'].astype(str) + rfm['f_score'].astype(str) + rfm['m_score'].astype(str)

c1,c2,c3,c4 = st.columns(4)
with c1: st.markdown(f'<div class="metric"><p>Total Customers</p><h3>{len(rfm)}</h3></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="metric"><p>Avg LTV</p><h3>${rfm["monetary"].mean():,.0f}</h3></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="metric"><p>Avg Frequency</p><h3>{rfm["frequency"].mean():.1f}</h3></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="metric"><p>Avg Recency</p><h3>{rfm["recency"].mean():.0f} days</h3></div>', unsafe_allow_html=True)

st.divider()

t1, t2, t3, t4 = st.tabs(["📊 RFM", "🎯 Segments", "🔗 Metrics", "👥 Details"])

with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(rfm, x='recency', y='frequency', size='monetary', color='monetary', 
                        color_continuous_scale='Viridis', title="RFM Scatter: Recency vs Frequency")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.scatter(rfm, x='frequency', y='monetary', size='recency', color='recency',
                        color_continuous_scale='Plasma', title="RFM Scatter: Frequency vs Monetary")
        st.plotly_chart(fig, use_container_width=True)

with t2:
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['segment'] = kmeans.fit_predict(rfm_scaled)
    
    segment_names = {0: 'VIP', 1: 'Loyal', 2: 'At Risk', 3: 'New'}
    rfm['segment_name'] = rfm['segment'].map(segment_names)
    
    c1, c2 = st.columns(2)
    with c1:
        seg_val = rfm.groupby('segment_name')['monetary'].sum().sort_values(ascending=False)
        fig = px.bar(x=seg_val.index, y=seg_val.values, color=seg_val.values, 
                    color_continuous_scale='Blues', title="Revenue by Segment")
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,.05)')
        st.plotly_chart(fig, use_container_width=True)
    
    with c2:
        seg_count = rfm['segment_name'].value_counts()
        fig = px.pie(values=seg_count.values, names=seg_count.index, title="Customers by Segment")
        st.plotly_chart(fig, use_container_width=True)

with t3:
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(rfm, x='recency', nbins=20, title="Recency Distribution", color_discrete_sequence=[C3])
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,.05)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(rfm, x='frequency', nbins=20, title="Frequency Distribution", color_discrete_sequence=[C4])
        fig.update_layout(height=400, plot_bgcolor='rgba(0,0,0,.05)')
        st.plotly_chart(fig, use_container_width=True)

with t4:
    display_df = rfm.reset_index()[['customer_id', 'recency', 'frequency', 'monetary', 'segment_name']].head(100)
    st.dataframe(display_df, use_container_width=True)
    csv = display_df.to_csv(index=False)
    st.download_button("📥 Download", csv, "segments.csv", "text/csv")
