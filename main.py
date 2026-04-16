"""Main analysis script for Customer Segmentation"""
import pandas as pd
from src.rfm_analysis import calculate_rfm, rfm_segmentation
from src.clustering import cluster_customers

def main():
    print("=" * 60)
    print("CUSTOMER SEGMENTATION & RFM ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/customer_transactions.csv')
    
    # Calculate RFM
    print("\n📊 Calculating RFM scores...")
    rfm = calculate_rfm(df)
    
    # RFM segmentation
    print("\n🏷️  RFM Segmentation:")
    rfm = rfm_segmentation(rfm)
    print(rfm[['recency', 'frequency', 'monetary', 'Segment']].head(10))
    
    # Segment summary
    print("\n📈 Customer Distribution by Segment:")
    print(rfm['Segment'].value_counts())
    
    # Clustering
    print("\n🎯 K-Means Clustering (k=3):")
    rfm, kmeans = cluster_customers(rfm, n_clusters=3)
    
    print("\nCluster Summary:")
    cluster_summary = rfm.groupby('Cluster')[['recency', 'frequency', 'monetary']].mean()
    print(cluster_summary.round(2))
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
