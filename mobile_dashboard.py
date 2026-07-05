import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set up professional page layout
st.set_page_config(page_title="Global Mobile Analytics & Recommendation System", layout="wide")

st.title("📱 Mobile Product Segmentation & Recommendation System")
st.markdown("### End-to-End Machine Learning Pipeline | Developed with Python & Streamlit")
st.markdown("---")

# ==========================================
# CACHED DATA & ML PIPELINE
# ==========================================
@st.cache_data
def load_and_process_data():
    df = pd.read_csv('global_mobile_reviews.csv')
    df = df.drop_duplicates().dropna()
    
    encoder = LabelEncoder()
    df['brand_encoded'] = encoder.fit_transform(df['brand'])
    df['model_encoded'] = encoder.fit_transform(df['model'])
    
    features_to_scale = [
        'price_usd', 'rating', 'battery_life_rating', 
        'camera_rating', 'performance_rating', 
        'design_rating', 'display_rating'
    ]
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features_to_scale])
    
    # 4 Clusters: Budget, Lower Mid, Upper Mid, Premium (Market Quartile Strategy)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    
    cluster_summary = df.groupby('cluster')['price_usd'].mean().sort_values().reset_index()
    tier_mapping = {
        cluster_summary.loc[0, 'cluster']: 'Budget Categories',
        cluster_summary.loc[1, 'cluster']: 'Lower Mid-Range',
        cluster_summary.loc[2, 'cluster']: 'Upper Mid-Range',
        cluster_summary.loc[3, 'cluster']: 'Premium Categories'
    }
    df['segment_name'] = df['cluster'].map(tier_mapping)
    
    # Clean Product Catalog
    product_catalog = df.groupby(['brand', 'model'])[features_to_scale].mean().reset_index()
    segments = df.groupby(['brand', 'model'])['segment_name'].agg(lambda x: x.mode()[0]).reset_index()
    product_catalog = pd.merge(product_catalog, segments, on=['brand', 'model'])
    
    catalog_scaled = scaler.transform(product_catalog[features_to_scale])
    
    return df, product_catalog, catalog_scaled, features_to_scale

df, product_catalog, catalog_scaled, features_to_scale = load_and_process_data()

# Sidebar Information
st.sidebar.header("📊 Dataset Summary Metrics")
st.sidebar.metric("Total Reviews Cleaned", f"{df.shape[0]:,}")
st.sidebar.metric("Unique Phone Models", f"{product_catalog.shape[0]}")
st.sidebar.markdown("---")
st.sidebar.success("✅ **Plotly Integration Active**")
st.sidebar.success("✅ **Price-Locked Recommendations Active**")
st.sidebar.success("✅ **Feature-Aware Suggestions Active**")

tab1, tab2, tab3 = st.tabs(["🎯 Market Segmentation Insights", "📈 Exploratory Visualizations", "🤖 Intelligent Recommendation System"])

# ==========================================
# TAB 1: MARKET SEGMENTATION (PLOTLY)
# ==========================================
with tab1:
    st.header("Cluster Interpretation & Business Segments")
    st.markdown("> **Market Quartile Strategy:** 4 clusters were strategically chosen to represent the natural economic purchasing tiers (Budget to Premium).")
    
    segment_metrics = df.groupby('segment_name').agg(
        Average_Price=('price_usd', 'mean'),
        Average_Rating=('rating', 'mean'),
        Total_Inventory=('model', 'count')
    ).reset_index()
    
    segment_metrics['Average_Price'] = segment_metrics['Average_Price'].map("${:,.2f}".format)
    segment_metrics['Average_Rating'] = segment_metrics['Average_Rating'].map("{:,.2f} / 5.0".format)
    st.table(segment_metrics)
    
    # Interactive Plotly Scatter Plot
    fig_cluster = px.scatter(
        df, x='price_usd', y='rating', color='segment_name', 
        hover_data=['brand'], opacity=0.6,
        title="Interactive Product Distribution Across Segments",
        labels={'price_usd': 'Price (USD)', 'rating': 'Overall Rating', 'segment_name': 'Market Tier'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

# ==========================================
# TAB 2: EXPLORATORY VISUALIZATIONS (PLOTLY)
# ==========================================
with tab2:
    st.header("Data Distribution & Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        top_brands = df['brand'].value_counts().head(10).reset_index()
        top_brands.columns = ['Brand', 'Total Reviews']
        fig_bar = px.bar(
            top_brands, x='Total Reviews', y='Brand', orientation='h',
            title="Top 10 Brand Penetration", color='Total Reviews', color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col2:
        fig_perf = px.scatter(
            df, x='price_usd', y='performance_rating', color='rating',
            title="Price vs. Performance Matrix", opacity=0.5,
            labels={'price_usd': 'Price (USD)', 'performance_rating': 'Hardware Performance'},
            color_continuous_scale='Magma'
        )
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature Correlation Heatmap")
    corr = df[features_to_scale].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale='RdBu_r', title="Hardware Spec Correlation"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ==========================================
# TAB 3: INTELLIGENT RECOMMENDATION SYSTEM
# ==========================================
with tab3:
    st.header("Intelligent Price-Locked Recommendation Engine")
    st.write("This engine isolates alternatives within the **same price tier** and analyzes hardware advantages.")
    
    available_models = sorted(product_catalog['model'].unique())
    selected_model = st.selectbox("Choose a Target Mobile Model:", available_models)
    
    if st.button("Generate Intelligent Recommendations"):
        target_idx = product_catalog[product_catalog['model'] == selected_model].index[0]
        target_data = product_catalog.iloc[target_idx]
        target_tier = target_data['segment_name']
        
        # 1. PRICE LOCKING: Filter catalog to only include phones in the exact same market tier
        tier_mask = product_catalog['segment_name'] == target_tier
        filtered_catalog = product_catalog[tier_mask].copy()
        
        # Get the new indices for our filtered dataset
        filtered_indices = filtered_catalog.index.tolist()
        
        # Generate vectors based only on the filtered subset
        filtered_vectors = catalog_scaled[filtered_indices]
        target_vector = catalog_scaled[target_idx].reshape(1, -1)
        
        # 2. CALCULATE SIMILARITY
        similarity_scores = cosine_similarity(target_vector, filtered_vectors).flatten()
        
        # Sort and get top 5 (ignoring the target phone itself which will be 100%)
        top_5_relative_idx = np.argsort(similarity_scores)[::-1]
        
        final_recommendations = []
        
        for rel_idx in top_5_relative_idx:
            actual_idx = filtered_indices[rel_idx]
            if actual_idx == target_idx:
                continue # Skip the exact same phone
            
            if len(final_recommendations) >= 5:
                break
                
            rec_data = product_catalog.iloc[actual_idx]
            match_pct = similarity_scores[rel_idx] * 100
            
            # 3. FEATURE ADVANTAGE LOGIC (Why buy this?)
            advantages = []
            if rec_data['camera_rating'] > target_data['camera_rating']:
                advantages.append("📸 Better Camera")
            if rec_data['battery_life_rating'] > target_data['battery_life_rating']:
                advantages.append("🔋 Longer Battery")
            if rec_data['performance_rating'] > target_data['performance_rating']:
                advantages.append("⚡ Faster Performance")
            
            why_buy = " | ".join(advantages) if advantages else "⚖️ Similar Specs"
            
            final_recommendations.append({
                'Brand': rec_data['brand'],
                'Model Name': rec_data['model'],
                'Price (USD)': f"${rec_data['price_usd']:,.2f}",
                'Market Tier': rec_data['segment_name'],
                'Hardware Advantage': why_buy,
                'Match Accuracy': f"{match_pct:.1f}%"
            })
            
        st.subheader(f"Top 5 Alternatives for '{selected_model}' (Locked to {target_tier}):")
        if final_recommendations:
            recs_df = pd.DataFrame(final_recommendations)
            st.dataframe(recs_df, use_container_width=True)
        else:
            st.warning("No other phones found in this specific price tier.")