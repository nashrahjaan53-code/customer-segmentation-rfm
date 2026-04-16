"""RFM Analysis module"""
import pandas as pd
from datetime import datetime, timedelta

def calculate_rfm(df, reference_date=None):
    """Calculate RFM scores for customers"""
    if reference_date is None:
        reference_date = df['purchase_date'].max() + timedelta(days=1)
    
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    rfm = df.groupby('customer_id').agg({
        'purchase_date': lambda x: (reference_date - x.max()).days,
        'transaction_id': 'count',
        'amount': 'sum'
    }).rename(columns={
        'purchase_date': 'recency',
        'transaction_id': 'frequency',
        'amount': 'monetary'
    })
    
    return rfm

def rfm_segmentation(rfm):
    """Create RFM segments"""
    rfm['R_score'] = pd.qcut(rfm['recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4], duplicates='drop')
    rfm['M_score'] = pd.qcut(rfm['monetary'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
    
    rfm['RFM_Segment'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    
    # Create meaningful labels
    rfm['Segment'] = rfm['RFM_Segment'].apply(label_segment)
    
    return rfm

def label_segment(rfm_score):
    """Label RFM segments meaningfully"""
    if rfm_score in ['444', '443', '434', '344']:
        return 'Champions'
    elif rfm_score in ['333', '334', '343', '433']:
        return 'Loyal'
    elif rfm_score in ['111', '112', '122', '211']:
        return 'Lost'
    else:
        return 'At-Risk'
