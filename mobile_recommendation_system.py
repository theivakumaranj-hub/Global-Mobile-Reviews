import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==========================================
# PHASE 1: DATA PREPROCESSING 
# ==========================================
print("Starting Phase 1: Data Preprocessing...")
df = pd.read_csv('global_mobile_reviews.csv')

# Clean data: drop duplicates and nulls
df = df.drop_duplicates().dropna()

# Encode categorical text into numbers for the machine learning model
encoder = LabelEncoder()
df['brand_encoded'] = encoder.fit_transform(df['brand'])
df['model_encoded'] = encoder.fit_transform(df['model'])

features_to_scale = [
    'price_usd', 'rating', 'battery_life_rating', 
    'camera_rating', 'performance_rating', 
    'design_rating', 'display_rating'
]

# Scale the features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features_to_scale]), columns=features_to_scale)

print(f"Phase 1 Complete! Data Shape ready for ML: {df_scaled.shape}\n")

# ==========================================
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("Starting Phase 2: Generating EDA Visualizations...")
sns.set_theme(style="whitegrid")

# 1. Product distribution across different brands 
plt.figure(figsize=(10, 6))
top_brands = df['brand'].value_counts().index[:10]
sns.countplot(y='brand', data=df, order=top_brands, palette='viridis')
plt.title('Top 10 Brands by Number of Reviews')
plt.xlabel('Total Reviews')
plt.ylabel('Brand')
plt.tight_layout()
plt.show() 

# 2. Relationships between price, ratings, and specifications (Correlation Matrix)
plt.figure(figsize=(10, 8))
correlation_matrix = df[features_to_scale].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix: Price, Ratings & Specs')
plt.tight_layout()
plt.show()

# 3. Price vs Performance Trend
plt.figure(figsize=(10, 6))
sns.scatterplot(x='price_usd', y='performance_rating', hue='rating', data=df, palette='magma', alpha=0.6)
plt.title('Price (USD) vs Performance Rating')
plt.xlabel('Price (USD)')
plt.ylabel('Performance Rating (1-5)')
plt.tight_layout()
plt.show()

print("Phase 2 Complete! Save these charts for your final EDA report.")

# ==========================================
# PHASE 3: CLUSTERING (PRODUCT SEGMENTATION)
# ==========================================
print("\nStarting Phase 3: K-Means Clustering...")

kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)
print("Clustering completed successfully!")

cluster_summary = df.groupby('cluster')[['price_usd', 'rating']].mean().reset_index()
cluster_summary = cluster_summary.sort_values(by='price_usd').reset_index(drop=True)

tier_mapping = {
    cluster_summary.loc[0, 'cluster']: 'Budget Categories',
    cluster_summary.loc[1, 'cluster']: 'Lower Mid-Range',
    cluster_summary.loc[2, 'cluster']: 'Upper Mid-Range',
    cluster_summary.loc[3, 'cluster']: 'Premium Categories'
}

df['segment_name'] = df['cluster'].map(tier_mapping)

print("\n--- Cluster Analysis Summary & Mean Metrics ---")
for index, row in cluster_summary.iterrows():
    orig_cluster = int(row['cluster'])
    assigned_tier = tier_mapping[orig_cluster]
    print(f"Cluster {orig_cluster} -> {assigned_tier}: Mean Price = ${row['price_usd']:.2f}, Mean Rating = {row['rating']:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(x='price_usd', y='rating', hue='segment_name', data=df, palette='Set1', alpha=0.7)
plt.title('Product Market Segmentation using K-Means (4 Clusters)')
plt.xlabel('Price (USD)')
plt.ylabel('Overall Rating')
plt.legend(title='Market Segment')
plt.tight_layout()
plt.show()

print("Phase 3 Complete! Save this segment chart for your report.")

# ==========================================
# PHASE 4: RECOMMENDATION SYSTEM DEVELOPMENT
# ==========================================
print("\nStarting Phase 4: Recommendation System Development...")

# FIXED CONTRADICTION: Group strictly by brand and model to avoid row fragmentation
product_catalog = df.groupby(['brand', 'model'])[features_to_scale].mean().reset_index()
segments = df.groupby(['brand', 'model'])['segment_name'].agg(lambda x: x.mode()[0]).reset_index()
product_catalog = pd.merge(product_catalog, segments, on=['brand', 'model'])

print(f"Unique product catalog created with {product_catalog.shape[0]} individual phone models.")

# Scale the aggregated features of our clean catalog using the previously fitted scaler
catalog_scaled = scaler.transform(product_catalog[features_to_scale])

# FIXED CONTRADICTION: Upgraded engine to handle price-locking and features to align with mobile_dashboard.py
def recommend_similar_mobiles(target_model_name, top_n=5):
    """
    Takes a phone model name, locks search to its specific price tier, 
    calculates cosine similarity, and highlights core hardware advantages.
    """
    model_mask = product_catalog['model'].str.lower() == target_model_name.lower()
    model_idx_list = product_catalog[model_mask].index
    
    if len(model_idx_list) == 0:
        return f"Model '{target_model_name}' not found in the database."
    
    target_idx = model_idx_list[0]
    target_data = product_catalog.iloc[target_idx]
    target_tier = target_data['segment_name']
    
    # Apply Price Locking to look only within the same cluster
    tier_mask = product_catalog['segment_name'] == target_tier
    filtered_catalog = product_catalog[tier_mask].copy()
    filtered_indices = filtered_catalog.index.tolist()
    
    filtered_vectors = catalog_scaled[filtered_indices]
    target_vector = catalog_scaled[target_idx].reshape(1, -1)
    
    # Calculate Cosine Similarity against filtered partition
    similarity_scores = cosine_similarity(target_vector, filtered_vectors).flatten()
    top_indices_relative = np.argsort(similarity_scores)[::-1]
    
    final_recommendations = []
    for rel_idx in top_indices_relative:
        actual_idx = filtered_indices[rel_idx]
        if actual_idx == target_idx:
            continue # Skip self
            
        if len(final_recommendations) >= top_n:
            break
            
        rec_data = product_catalog.iloc[actual_idx]
        match_pct = similarity_scores[rel_idx] * 100
        
        # Formulate feature advantages
        advantages = []
        if rec_data['camera_rating'] > target_data['camera_rating']:
            advantages.append("Better Camera")
        if rec_data['battery_life_rating'] > target_data['battery_life_rating']:
            advantages.append("Longer Battery")
        if rec_data['performance_rating'] > target_data['performance_rating']:
            advantages.append("Faster Performance")
        
        why_buy = " | ".join(advantages) if advantages else "Similar Specs"
        
        final_recommendations.append({
            'brand': rec_data['brand'],
            'model': rec_data['model'],
            'segment_name': rec_data['segment_name'],
            'price_usd': f"${rec_data['price_usd']:,.2f}",
            'rating': round(rec_data['rating'], 2),
            'hardware_advantage': why_buy,
            'similarity_match_%': f"{match_pct:.1f}%"
        })
        
    return pd.DataFrame(final_recommendations)

# Validate recommendation relevance (Testing the engine)
print("\n--- Testing Recommendation Engine Relevance ---")
sample_phone = product_catalog.iloc[0]['model']
print(f"Generating top 5 recommendations for a user looking at: '{sample_phone}'")

test_recommendations = recommend_similar_mobiles(sample_phone, top_n=5)
print(test_recommendations.to_string(index=False))

print("\nPhase 4 Complete! Recommendation engine validated and operational.")