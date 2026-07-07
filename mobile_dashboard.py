import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Set up professional page layout at the absolute top
st.set_page_config(page_title="Global Mobile Analytics & Recommendation System", layout="wide")

st.title("📱 Mobile Product Segmentation & Recommendation System")
st.markdown("### End-to-End Machine Learning Pipeline | Developed with Python & Streamlit")
st.markdown("---")

# 1. DIRECT FILE CHECK: Intercept missing dataset errors before they cause a server crash
if not os.path.exists('global_mobile_reviews.csv'):
    st.error("🚨 **Critical File Error: Dataset Missing!**")
    st.markdown(f"The system cannot locate `global_mobile_reviews.csv` in your active directory: `{os.getcwd()}`")
    st.write("Please confirm that the dataset CSV file is spelled correctly and sits in the exact same folder as this script.")
    st.stop()

# 2. CACHE PURGED: Stripped @st.cache_data to force raw errors to bubble up instantly
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
    
    product_catalog = df.groupby(['brand', 'model'])[features_to_scale].mean().reset_index()
    segments = df.groupby(['brand', 'model'])['segment_name'].agg(lambda x: x.mode()[0]).reset_index()
    product_catalog = pd.merge(product_catalog, segments, on=['brand', 'model'])
    
    catalog_scaled = scaler.transform(product_catalog[features_to_scale])
    
    return df, product_catalog, catalog_scaled, features_to_scale

# Execute pipeline safely with a local error catch
try:
    df, product_catalog, catalog_scaled, features_to_scale = load_and_process_data()
except Exception as pipeline_error:
    st.error(f"🚨 **Data Processing Exception Matrix:** {pipeline_error}")
    st.stop()

# Sidebar Information
st.sidebar.header("📊 Dataset Summary Metrics")
st.sidebar.metric("Total Reviews Cleaned", f"{df.shape[0]:,}")
st.sidebar.metric("Unique Phone Models", f"{product_catalog.shape[0]}")
st.sidebar.markdown("---")
st.sidebar.success("✅ **Plotly Integration Active**")
st.sidebar.success("✅ **Price-Locked Recommendations Active**")
st.sidebar.success("✅ **Smart Consumer Controls Active**")

# Interactive Tab Navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Market Segmentation Insights", 
    "📈 Exploratory Visualizations", 
    "🤖 Intelligent Recommendation System",
    "💡 Smart Buyer's Assistant"
])

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
        
        tier_mask = product_catalog['segment_name'] == target_tier
        filtered_catalog = product_catalog[tier_mask].copy()
        filtered_indices = filtered_catalog.index.tolist()
        
        filtered_vectors = catalog_scaled[filtered_indices]
        target_vector = catalog_scaled[target_idx].reshape(1, -1)
        
        similarity_scores = cosine_similarity(target_vector, filtered_vectors).flatten()
        top_5_relative_idx = np.argsort(similarity_scores)[::-1]
        
        final_recommendations = []
        for rel_idx in top_5_relative_idx:
            actual_idx = filtered_indices[rel_idx]
            if actual_idx == target_idx:
                continue
            
            if len(final_recommendations) >= 5:
                break
                
            rec_data = product_catalog.iloc[actual_idx]
            match_pct = similarity_scores[rel_idx] * 100
            
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

# ==========================================
# TAB 4: SMART BUYER'S ASSISTANT
# ==========================================
with tab4:
    st.header("💡 Smart Buyer's Value Optimizer")
    st.write("Input your budget ceiling and choose your priority feature to instantly extract the highest-performing models.")
    
    min_b = int(product_catalog['price_usd'].min())
    max_b = int(product_catalog['price_usd'].max())
    if min_b >= max_b:
        max_b = min_b + 100
    mid_b = int((min_b + max_b) / 2)
    
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        max_budget = st.slider(
            "Select Your Maximum Budget (USD):", 
            min_value=min_b, 
            max_value=max_b, 
            value=mid_b, 
            step=25
        )
    with col_input2:
        priority_feature = st.selectbox(
            "Select Your Priority Hardware Focus:",
            options=['battery_life_rating', 'camera_rating', 'performance_rating', 'design_rating', 'display_rating'],
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
    filtered_buyer_df = product_catalog[product_catalog['price_usd'] <= max_budget].copy()
    
    if not filtered_buyer_df.empty:
        top_matches = filtered_buyer_df.sort_values(by=[priority_feature, 'rating'], ascending=False).head(5)
        
        fig_buyer = px.bar(
            top_matches,
            x=priority_feature,
            y='model',
            color='price_usd',
            orientation='h',
            title=f"Top Value Models Under ${max_budget} Optimizing {priority_feature.replace('_', ' ').title()}",
            labels={'model': 'Phone Model', priority_feature: 'Hardware Rating', 'price_usd': 'Cost (USD)'},
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_buyer, use_container_width=True)
        
        st.subheader("📊 Specifications Breakdown for Top Value Picks")
        display_buyer_df = top_matches.copy().rename(columns={
            'brand': 'Brand', 'model': 'Model Name', 'price_usd': 'Price (USD)', 
            'rating': 'Overall Rating', priority_feature: priority_feature.replace('_', ' ').title()
        })
        
        display_buyer_df['Price (USD)'] = display_buyer_df['Price (USD)'].map("${:,.2f}".format)
        display_buyer_df['Overall Rating'] = display_buyer_df['Overall Rating'].map("{:,.2f} / 5.0".format)
        
        show_cols = ['Brand', 'Model Name', 'Price (USD)', 'Overall Rating', priority_feature.replace('_', ' ').title(), 'segment_name']
        st.dataframe(display_buyer_df[show_cols], use_container_width=True)
    else:
        st.warning("No unique phone models detected below your selected budget ceiling. Try increasing the slider value.")
