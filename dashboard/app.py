import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.rfm_analysis import calculate_rfm, rfm_segmentation
from src.clustering import cluster_customers

st.set_page_config(page_title="Customer Segmentation", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data
df = pd.read_csv('data/customer_transactions.csv')

if df is not None and len(df) > 0:
    st.markdown('<div class="header"><h1>👥 Customer Segmentation & RFM Analysis Dashboard</h1></div>', 
                unsafe_allow_html=True)
    
    # Calculate RFM
    rfm = calculate_rfm(df)
    rfm = rfm_segmentation(rfm)
    rfm_cluster, kmeans = cluster_customers(rfm, n_clusters=3)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(rfm))
    with col2:
        st.metric("Champions", len(rfm[rfm['Segment'] == 'Champions']))
    with col3:
        st.metric("At-Risk", len(rfm[rfm['Segment'] == 'At-Risk']))
    with col4:
        st.metric("Avg Monetary", f"${rfm['monetary'].mean():,.0f}")
    
    st.divider()
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 RFM Overview", "🎯 Segmentation", "🔍 Clustering", "📈 Heatmap"])
    
    # TAB 1: RFM OVERVIEW
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Recency Distribution (Histogram)")
            fig_recency = px.histogram(rfm, x='recency', nbins=10,
                                      title="Days Since Purchase Distribution",
                                      color_discrete_sequence=['#f093fb'])
            st.plotly_chart(fig_recency, use_container_width=True)
        
        with col2:
            st.subheader("Frequency Distribution (Bar)")
            freq_dist = rfm['frequency'].value_counts().sort_index()
            fig_freq = px.bar(x=freq_dist.index, y=freq_dist.values,
                            title="Purchase Frequency Distribution",
                            labels={'x': 'Frequency', 'y': 'Count'},
                            color_discrete_sequence=['#f5576c'])
            st.plotly_chart(fig_freq, use_container_width=True)
        
        with col3:
            st.subheader("Monetary Distribution (Box Plot)")
            fig_monetary = px.box(rfm, y='monetary',
                                 title="Monetary Value Distribution",
                                 color_discrete_sequence=['#4facfe'])
            st.plotly_chart(fig_monetary, use_container_width=True)
        
        # 3D Scatter
        st.subheader("RFM 3D Scatter Plot")
        fig_3d = px.scatter_3d(rfm.reset_index(), x='recency', y='frequency', z='monetary',
                             color='Segment', title="3D Customer RFM Space",
                             color_discrete_map={'Champions': '#f093fb', 'Loyal': '#4facfe', 
                                                'At-Risk': '#f5576c', 'Lost': '#808080'})
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # TAB 2: SEGMENTATION
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Customer Segments Pie Chart")
            segment_counts = rfm['Segment'].value_counts()
            fig_seg_pie = px.pie(values=segment_counts.values, names=segment_counts.index,
                               title="Customer Distribution by Segment",
                               color_discrete_map={'Champions': '#f093fb', 'Loyal': '#4facfe',
                                                  'At-Risk': '#f5576c', 'Lost': '#808080'})
            st.plotly_chart(fig_seg_pie, use_container_width=True)
        
        with col2:
            st.subheader("Segment Metrics Comparison")
            segment_metrics = rfm.groupby('Segment')[['recency', 'frequency', 'monetary']].mean()
            fig_radar = go.Figure()
            
            for segment in segment_metrics.index:
                fig_radar.add_trace(go.Scatterpolar(
                    r=segment_metrics.loc[segment].values,
                    theta=['Recency', 'Frequency', 'Monetary'],
                    fill='toself',
                    name=segment
                ))
            
            fig_radar.update_layout(title="Segment Characteristics (Radar Chart)")
            st.plotly_chart(fig_radar, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Average Revenue by Segment (Bar)")
            segment_revenue = rfm.groupby('Segment')['monetary'].mean().sort_values(ascending=False)
            fig_seg_bar = px.bar(x=segment_revenue.index, y=segment_revenue.values,
                               title="Average Customer Value by Segment",
                               labels={'x': 'Segment', 'y': 'Avg Monetary ($)'},
                               color=segment_revenue.index,
                               color_discrete_map={'Champions': '#f093fb', 'Loyal': '#4facfe',
                                                  'At-Risk': '#f5576c', 'Lost': '#808080'})
            st.plotly_chart(fig_seg_bar, use_container_width=True)
        
        with col2:
            st.subheader("Segment Size Comparison (Bar)")
            size_data = rfm['Segment'].value_counts().sort_values(ascending=False)
            fig_size = px.bar(x=size_data.index, y=size_data.values,
                            title="Number of Customers per Segment",
                            labels={'x': 'Segment', 'y': 'Count'},
                            color=size_data.index,
                            color_discrete_map={'Champions': '#f093fb', 'Loyal': '#4facfe',
                                               'At-Risk': '#f5576c', 'Lost': '#808080'})
            st.plotly_chart(fig_size, use_container_width=True)
    
    # TAB 3: CLUSTERING
    with tab3:
        st.subheader("K-Means Clustering Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Distribution (Pie)")
            cluster_counts = rfm_cluster['Cluster'].value_counts()
            fig_cluster_pie = px.pie(values=cluster_counts.values, names=cluster_counts.index,
                                   title="Customers per Cluster",
                                   color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_cluster_pie, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Characteristics")
            cluster_stats = rfm_cluster.groupby('Cluster')[['recency', 'frequency', 'monetary']].mean()
            st.dataframe(cluster_stats.round(2), use_container_width=True)
        
        st.subheader("Scatter: Monetary vs Frequency (Colored by Cluster)")
        fig_scatter = px.scatter(rfm_cluster.reset_index(), x='frequency', y='monetary',
                               color='Cluster', size='recency',
                               title="Customer Clusters: Frequency vs Monetary Value",
                               hover_name='customer_id')
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # TAB 4: HEATMAP
    with tab4:
        st.subheader("Correlation Heatmap")
        
        corr_data = rfm[['recency', 'frequency', 'monetary']].corr()
        fig_heatmap = px.imshow(corr_data, text_auto=True, aspect='auto',
                              title="RFM Correlation Matrix",
                              color_continuous_scale='RdBu')
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("Customer Data Table")
        st.dataframe(rfm_cluster.reset_index(), use_container_width=True, hide_index=True)
        
        # Download
        csv = rfm_cluster.reset_index().to_csv(index=False)
        st.download_button(
            label="📥 Download Segmentation Data",
            data=csv,
            file_name="customer_segments.csv",
            mime="text/csv"
        )

else:
    st.error("Unable to load data.")
