import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import ast
import re

# ✅ Set page config first
st.set_page_config(page_title="Beauty Review NLP Dashboard", layout="wide")

# ---------- Load Cleaned CSVs ----------
REVIEWS_PATH = "/Users/poojakabadi/Documents/Documents/UB/3. Spring/MS Practium/cleaned_data/merged_reviews_with_sentiment.csv"
PRODUCTS_PATH = "/Users/poojakabadi/Documents/Documents/UB/3. Spring/MS Practium/cleaned_data/product.csv"

@st.cache_data
def load_data():
    reviews = pd.read_csv(REVIEWS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)

    reviews['product_name'] = reviews['product_name_x']
    reviews['brand_name'] = reviews['brand_name_x'].combine_first(reviews['brand_name_y'])
    reviews['rating'] = reviews['rating_x']

    drop_cols = ['product_name_x', 'product_name_y', 'brand_name_x', 'brand_name_y', 'rating_x', 'rating_y']
    reviews.drop(columns=[col for col in drop_cols if col in reviews.columns], inplace=True)

    return reviews, products

reviews, products = load_data()
reviews['submission_time'] = pd.to_datetime(reviews['submission_time'], errors='coerce')

# ---------- Sidebar Filters ----------
st.sidebar.title("🔍 Filters")
min_date, max_date = reviews['submission_time'].min(), reviews['submission_time'].max()
start_date, end_date = st.sidebar.date_input("Filter by date range:", [min_date, max_date])
mask = (reviews['submission_time'] >= pd.to_datetime(start_date)) & (reviews['submission_time'] <= pd.to_datetime(end_date))
filtered = reviews.loc[mask]

brands = filtered['brand_name'].dropna().unique()
selected_brand = st.sidebar.selectbox("Select Brand", ["All"] + sorted(list(brands)))
if selected_brand != "All":
    filtered = filtered[filtered['brand_name'] == selected_brand]

products_list = filtered['product_name'].dropna().unique()
selected_product = st.sidebar.selectbox("Select Product", ["All"] + sorted(list(products_list)))
if selected_product != "All":
    filtered = filtered[filtered['product_name'] == selected_product]

# ---------- Tabs ----------
st.title("Beauty Intelligence Hub: Know Your Product Before You Glow")
tabs = st.tabs(["📊 Summary", "📝 Sentiment", "🌟 Ratings","🧪 Ingredient Risk Checker", "💬 Word Cloud"])

# ---------- Summary ----------
with tabs[0]:
    st.markdown("### 📊 Combined Summary", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Total Reviews", f"{len(filtered):,}")
    col2.metric("🧴 Unique Products", f"{filtered['product_name'].nunique():,}")
    col3.metric("🏷️ Brands", f"{filtered['brand_name'].nunique():,}")

    rating_counts = filtered['rating'].value_counts().sort_index()
    fig1 = px.bar(x=rating_counts.index, y=rating_counts.values,
                  labels={'x': 'Rating', 'y': 'Number of Reviews'},
                  color_discrete_sequence=["#0A2F63"], height=400)
    fig1.update_layout(title="Rating Distribution", title_font_size=20)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### ⏱ Monthly Review Submission Trend")
    if 'submission_time' in filtered.columns:
        monthly = filtered.dropna(subset=['submission_time'])
        monthly = monthly['submission_time'].dt.to_period('M').value_counts().sort_index()
        monthly.index = monthly.index.to_timestamp()
        fig2 = px.line(x=monthly.index, y=monthly.values,
                       labels={'x': 'Month', 'y': 'Number of Reviews'},
                       markers=True, color_discrete_sequence=["#0A2F63"])
        fig2.update_layout(title="Monthly Review Trends", title_font_size=20)
        st.plotly_chart(fig2, use_container_width=True)

    top_products = filtered['product_name'].value_counts().nlargest(10)
    fig3 = px.bar(y=top_products.index[::-1], x=top_products.values[::-1],
                  labels={'x': 'Review Count', 'y': 'Product Name'},
                  orientation='h',
                  color=top_products.values[::-1],
                  color_continuous_scale=px.colors.sequential.Blues)
    fig3.update_layout(title="Top Reviewed Products", title_font_size=20, coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    avg_rating = filtered[filtered['product_name'].isin(top_products.index)] \
                    .groupby('product_name')['rating'].mean().sort_values()
    fig4 = px.bar(x=avg_rating.values, y=avg_rating.index,
                  orientation='h',
                  labels={'x': 'Average Rating', 'y': 'Product'},
                  color=avg_rating.values,
                  color_continuous_scale=px.colors.sequential.Blues)
    fig4.update_layout(title="Avg Rating of Top Products", title_font_size=20, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


# ---------- Sentiment Tab ----------
with tabs[1]:
    st.markdown("### 📝 Sentiment Analysis")

    if "sentiment" not in filtered.columns:
        st.error("❌ Sentiment column not found. Please use the dataset with sentiment.")
    elif filtered['sentiment'].dropna().empty:
        st.warning("⚠️ No sentiment data available in the filtered dataset.")
    else:
        # Convert sentiment score to label
        def classify_sentiment(score):
            if score >= 0.2:
                return "Positive"
            elif score <= -0.2:
                return "Negative"
            else:
                return "Neutral"

        filtered['sentiment_label'] = filtered['sentiment'].apply(classify_sentiment)
        sentiment_counts = filtered['sentiment_label'].value_counts()

        # Donut Chart
        fig_sentiment = px.pie(
            names=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.5,
            title="Sentiment Distribution",
            color=sentiment_counts.index,
            color_discrete_map={"Positive": "#172d60", "Neutral": "#4d88c2", "Negative": "#8bb9d7"}
        )
        st.plotly_chart(fig_sentiment, use_container_width=True)

        # Sample Reviews
        with st.expander("📋 Sample Reviews", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 👍 Positive Reviews")
                for i, r in enumerate(filtered[filtered['sentiment_label'] == "Positive"]['review_text'].dropna().head(3), 1):
                    st.markdown(f"**{i}.** {r}")
            with col2:
                st.markdown("#### 👎 Negative Reviews")
                for i, r in enumerate(filtered[filtered['sentiment_label'] == "Negative"]['review_text'].dropna().head(3), 1):
                    st.markdown(f"**{i}.** {r}")

        # Buzzwords by Sentiment
        from sklearn.feature_extraction.text import CountVectorizer
        import string
        from nltk.corpus import stopwords
        import nltk

        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            # Lowercase
            text = text.lower()
            # Remove punctuation and digits
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\d+', '', text)
            # Remove stopwords
            words = [word for word in text.split() if word not in stop_words]
            return ' '.join(words)

        def top_words_by_sentiment(df, label, top_n=5):
            texts = df[df['sentiment_label'] == label]['review_text'].dropna().apply(clean_text)
            if texts.empty:
                return pd.Series()
            vec = CountVectorizer(max_features=1000)
            X = vec.fit_transform(texts)
            word_counts = X.sum(axis=0).A1
            return pd.Series(word_counts, index=vec.get_feature_names_out()).nlargest(top_n)

        st.markdown("### 🔍 Top Buzzwords by Sentiment")
        for label, color in zip(["Positive", "Neutral", "Negative"], ["#172d60", "#4d88c2","#8bb9d7" ]):
            buzzwords = top_words_by_sentiment(filtered, label)
            if not buzzwords.empty:
                fig = px.bar(
                    x=buzzwords.values,
                    y=buzzwords.index,
                    orientation='h',
                    title=f"Top Words in {label} Reviews",
                    labels={"x": "Count", "y": "Word"},
                    color_discrete_sequence=[color]
                )
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                st.plotly_chart(fig, use_container_width=True)

        # Sentiment Score Distribution
        st.markdown("### 📊 Sentiment Score Distribution")
        fig_hist = px.histogram(
            filtered.dropna(subset=['sentiment']),
            x='sentiment',
            nbins=30,
            title="Distribution of Sentiment Scores",
            color_discrete_sequence=["#0A2F63"]
        )
        fig_hist.update_layout(bargap=0.1)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Average Sentiment by Brand
        st.markdown("### 🏷️ Average Sentiment Score by Brand (Top 10)")
        avg_brand = filtered.groupby("brand_name")["sentiment"].mean().sort_values(ascending=True).head(10)
        fig_brand = px.bar(
            x=avg_brand.values,
            y=avg_brand.index,
            orientation='h',
            labels={'x': 'Average Sentiment Score', 'y': 'Brand'},
            color=avg_brand.values,
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig_brand.update_layout(title="Top 10 Brands by Avg Sentiment", coloraxis_showscale=False)
        st.plotly_chart(fig_brand, use_container_width=True)

# ---------- Ratings Tab ----------
with tabs[2]:
    st.markdown("### 🌟 Ratings Insights")

    # Filters (Optional Skin Type / Skin Concern if columns exist)
    if 'skin_type' in filtered.columns:
        skin_types = filtered['skin_type'].dropna().unique()
        selected_skin = st.selectbox("Filter by Skin Type", ["All"] + sorted(skin_types))
        if selected_skin != "All":
            filtered = filtered[filtered['skin_type'] == selected_skin]

    if 'skin_concerns' in filtered.columns:
        concerns = filtered['skin_concerns'].dropna().unique()
        selected_concern = st.selectbox("Filter by Skin Concern", ["All"] + sorted(concerns))
        if selected_concern != "All":
            filtered = filtered[filtered['skin_concerns'] == selected_concern]

    # ⭐ Rating Breakdown by Star Levels
    st.markdown("#### ⭐ Rating Breakdown")
    rating_counts = filtered['rating'].value_counts().sort_index()
    fig1 = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={'x': 'Rating (Stars)', 'y': 'Number of Reviews'},
        color_discrete_sequence=["#0A2F63"]
    )
    fig1.update_layout(title="Review Rating Distribution", title_font_size=18)
    st.plotly_chart(fig1, use_container_width=True)

    # 🏆 Top Rated Brands by Category (descending)
    st.markdown("#### 🏆 Top Rated Brands by Category")
    if 'primary_category' in filtered.columns:
        top_rated = filtered.groupby(['primary_category', 'brand_name'])['rating'].mean().reset_index()
        top_rated = top_rated.sort_values(by='rating', ascending=True).dropna()
        fig2 = px.bar(
            top_rated.head(15),
            x='rating',
            y='brand_name',
            color='primary_category',
            orientation='h',
            title="Top Rated Products by Category",
            labels={'rating': 'Average Rating', 'brand_name': 'Product'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Primary category column not found in dataset.")

    # ⏱ Rating Trend Over Time
    st.markdown("#### ⏱ Rating Trend Over Time")
    if 'submission_time' in filtered.columns:
        time_trend = filtered.dropna(subset=['submission_time'])
        time_trend['month'] = time_trend['submission_time'].dt.to_period("M").astype(str)
        trend_data = time_trend.groupby('month')['rating'].mean().reset_index()
        fig3 = px.line(
            trend_data,
            x='month',
            y='rating',
            markers=True,
            title="Average Rating Over Time",
            labels={'month': 'Month', 'rating': 'Avg Rating'},
            color_discrete_sequence=["#0A2F63"]
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Submission time column is missing.")

    # 🔁 Rating vs Sentiment Alignment
    st.markdown("#### 🔁 Rating vs Sentiment")
    if 'sentiment' in filtered.columns:
        sentiment_bins = pd.cut(filtered['rating'], bins=[0, 2, 3.5, 5], labels=['Low (0–2)', 'Medium (2–3.5)', 'High (3.5–5)'])
        alignment = filtered.groupby(sentiment_bins)['sentiment'].mean().reset_index()
        fig4 = px.bar(
            alignment,
            x='rating',
            y='sentiment',
            labels={'rating': 'Rating Group', 'sentiment': 'Avg Sentiment Score'},
            color='sentiment',
            color_continuous_scale='Blues',
            title="Sentiment Score by Rating Group"
        )
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Sentiment column not available to show alignment.")

# ---------- Ingredients Risk Checker ----------
with tabs[3]:
    st.subheader("🧪 Ingredient Risk Checker")

    carcinogens = {
        'formaldehyde', 'coal tar', 'benzene', 'quaternium-15', '1,4-dioxane',
        'lead acetate', 'polyethylene glycol (peg)', 'toluene', 'carbon black',
        'petrolatum', 'ethanolamine', 'diethanolamine', 'triethanolamine',
        'acrylamide', 'benzophenone', 'styrene', 'methylene chloride',
        'methoxyethanol', 'hydroquinone', 'p-phenylenediamine', 'isoeugenol',
        'retinyl palmitate', 'lead', 'cadmium', 'chromium', 'asbestos', 'talc (contaminated)'
    }

    allergens = {
        'fragrance', 'parfum', 'parabens', 'methylisothiazolinone', 'cocamidopropyl betaine',
        'lanolin', 'propylene glycol', 'bht', 'limonene', 'linalool', 'linalool.',
        'eugenol', 'citral', 'citral.', 'geraniol', 'geraniol.',
        'ammonium laureth sulfate', 'ammonium lauryl sulfate', 
        'sodium lauryl sulfate (sls)', 'sodium laureth sulfate',
        'benzyl salicylate', 'benzyl alcohol', 'benzyl benzoate',
        'benzyl cinnamate', 'farnesol', 'coumarin', 'alpha-isomethyl ionone',
        'hexyl cinnamal', 'cinnamal', 'cinnamyl alcohol', 'citronellol', 'isoeugenol'
    }

    endocrine_disruptors = {
        'triclosan', 'phthalates', 'dibutyl phthalate (dbp)', 'diethyl phthalate (dep)',
        'butylated hydroxyanisole (bha)', 'butylated hydroxytoluene (bht)', 
        'ethylhexyl methoxycinnamate', 'oxybenzone', 'octinoxate', 
        'homosalate', 'benzophenone-3', 'resorcinol', 'nonylphenol', 
        'ethylparaben', 'butylparaben', 'propylparaben',
        'butyl methoxydibenzoylmethane', 'ethylhexyl salicylate',
        'methylparaben', 'phenoxyethanol', 'avobenzone', 'octocrylene'
    }

    def analyze_ingredients(raw_ingredient_str):
        try:
            raw_list = ast.literal_eval(raw_ingredient_str)
        except Exception:
            return {'Carcinogens': [], 'Allergens': [], 'Endocrine Disruptors': [], 'Total Count': 0}
        cleaned = []
        for item in raw_list:
            item = re.sub(r"\(.*?\)", "", item).replace("/", " ").replace("*", "").lower().strip()
            cleaned.extend([i.strip() for i in item.split(",") if i.strip()])
        c = [i for i in cleaned if i in carcinogens]
        a = [i for i in cleaned if i in allergens]
        e = [i for i in cleaned if i in endocrine_disruptors]
        return {'Carcinogens': c, 'Allergens': a, 'Endocrine Disruptors': e, 'Total Count': len(set(c + a + e))}

    def get_risk_label(count):
        if count == 0:
            return "✅ Safe", "#4CAF50"  # Green
        elif count <= 2:
            return "⚠️ Moderate Risk", "#FFC107"  # Yellow
        else:
            return "🔥 High Risk", "#F44336"  # Red

    selected_row = products[products['product_name'] == selected_product] if selected_product != "All" else pd.DataFrame()
    if not selected_row.empty:
        ing = selected_row.iloc[0]['ingredients']
        result = analyze_ingredients(ing)
        risk_label, risk_color = get_risk_label(result['Total Count'])
        st.markdown(f"<div style='background-color:{risk_color};padding:1rem;border-radius:8px;width:fit-content;font-weight:bold'>Risk Level: {risk_label}</div>", unsafe_allow_html=True)
        st.markdown(f"**Total Harmful Ingredients Detected**: `{result['Total Count']}`")


        # ✅ Add definitions
        st.markdown("### What These Categories Mean")
        st.markdown("""
        - **☣️ Carcinogens**: Substances that may cause cancer.  
        - **🌸 Allergens**: Ingredients that may trigger allergic reactions or skin sensitivity.  
        - **🧬 Endocrine Disruptors**: Chemicals that can interfere with hormonal balance.
        """)
        
        for label, found, icon in [
            ("Carcinogens", result['Carcinogens'], "☣️"),
            ("Allergens", result['Allergens'], "🌸"),
            ("Endocrine Disruptors", result['Endocrine Disruptors'], "🧬")
        ]:
        
            st.markdown(f"#### {icon} {label}")
            if found:
                for i in found:
                    st.markdown(f"- **{i.title()}**")
            else:
                st.markdown("*None found*")
    else:
        st.info("Select a product from the sidebar to see its ingredients analysis.")

# ---------- Word Cloud Tab ----------
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import string

with tabs[4]:
    st.markdown("### 💬 Word Cloud of Customer Reviews")

    if 'review_text' not in filtered.columns or filtered['review_text'].dropna().empty:
        st.warning("⚠️ No review text available in the filtered dataset.")
    else:
        # Combine and clean all review text
        stop_words = set(STOPWORDS)
        all_reviews = filtered['review_text'].dropna().astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
        cleaned_reviews = " ".join(all_reviews)
        cleaned_reviews = " ".join([word for word in cleaned_reviews.split() if word not in stop_words])

        # Generate word cloud
        wordcloud = WordCloud(
            width=1000,
            height=500,
            background_color="#0A2F63",  # Dark blue
            colormap='Pastel1',
            stopwords=stop_words,
            max_words=100
        ).generate(cleaned_reviews)

        # Display word cloud
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

