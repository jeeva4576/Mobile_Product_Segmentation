#streamlit run Mobile_app.py

import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st


def cosine_similarity_manual(user_vector, product_vectors):
    user_vector = np.asarray(user_vector, dtype=float)
    product_vectors = np.asarray(product_vectors, dtype=float)

    user_norm = np.linalg.norm(user_vector)
    product_norms = np.linalg.norm(product_vectors, axis=1)

    return (product_vectors @ user_vector) / (product_norms * user_norm)


st.set_page_config(
    page_title="Mobile Product Recommendation System",
    page_icon="📱",
    layout="wide"
)


@st.cache_resource
def load_model_package():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "Output",
        "mobile_recommendation_model.pkl"
    )

    with open(model_path, "rb") as file:
        return pickle.load(file)


model_package = load_model_package()

product_df = model_package["product_df"]
product_normalized_features = model_package["product_normalized_features"]
brand_profiles = model_package["brand_profiles"]
price_min = model_package["price_min"]
price_max = model_package["price_max"]
product_features = model_package["product_features"]
rating_features = model_package["rating_features"]
weights = model_package["recommendation_weights"]


def get_recommendations(
    brand,
    price_usd,
    rating,
    performance_rating,
    camera_rating,
    battery_life_rating,
    display_rating,
    design_rating,
    top_n=10
):
    user_price_normalized = (
        (price_usd - price_min) / (price_max - price_min)
    )

    user_preferences = {
        "rating": rating,
        "performance_rating": performance_rating,
        "camera_rating": camera_rating,
        "battery_life_rating": battery_life_rating,
        "display_rating": display_rating,
        "design_rating": design_rating
    }

    user_rating_normalized = {}

    for feature in rating_features:
        user_rating_normalized[feature] = (
            user_preferences[feature] - 1
        ) / 4

    user_normalized = pd.DataFrame([{
        "price_usd": user_price_normalized,
        "rating": user_rating_normalized["rating"],
        "performance_rating": user_rating_normalized["performance_rating"],
        "camera_rating": user_rating_normalized["camera_rating"],
        "battery_life_rating": user_rating_normalized["battery_life_rating"],
        "display_rating": user_rating_normalized["display_rating"],
        "design_rating": user_rating_normalized["design_rating"]
    }])

    feature_similarity = pd.DataFrame(index=product_df.index)

    for feature in product_features:
        feature_similarity[feature] = (
            1 - abs(
                product_normalized_features[feature]
                - user_normalized.iloc[0][feature]
            )
        )

    cosine_scores = cosine_similarity_manual(
        user_normalized[product_features].iloc[0].values,
        product_normalized_features[product_features].values
    )

    selected_brand_profile = (
        brand_profiles.loc[brand].values.reshape(1, -1)
    )

    brand_cosine_scores = cosine_similarity_manual(
        selected_brand_profile[0],
        brand_profiles.values
    )

    brand_similarity_map = dict(
        zip(brand_profiles.index, brand_cosine_scores)
    )

    result = feature_similarity.copy()
    result["brand"] = product_df["brand"].values
    result["model"] = product_df["model"].values
    result["brand_similarity"] = result["brand"].map(
        brand_similarity_map
    )
    result["cosine_similarity"] = cosine_scores

    result["final_score"] = (
        result["price_usd"] * weights["price_usd"] +
        result["rating"] * weights["rating"] +
        result["performance_rating"] * weights["performance_rating"] +
        result["camera_rating"] * weights["camera_rating"] +
        result["battery_life_rating"] * weights["battery_life_rating"] +
        result["display_rating"] * weights["display_rating"] +
        result["design_rating"] * weights["design_rating"] +
        result["cosine_similarity"] * weights["cosine_similarity"] +
        result["brand_similarity"] * weights["brand_similarity"]
    )

    result = result.sort_values(
        "final_score",
        ascending=False
    ).head(top_n)

    result = result[["brand", "model", "final_score"]].merge(
        product_df,
        on=["brand", "model"],
        how="left"
    )

    return result[
        [
            "brand",
            "model",
            "price_usd",
            "rating",
            "performance_rating",
            "camera_rating",
            "battery_life_rating",
            "display_rating",
            "design_rating",
            "final_score"
        ]
    ].reset_index(drop=True)


st.title("📱 Mobile Product Recommendation System")

st.write(
    "Find mobile products based on your preferred brand, "
    "budget, overall rating, and product feature preferences."
)

st.divider()

st.subheader("Enter Your Preferences")

col1, col2 = st.columns(2)

with col1:
    selected_brand = st.selectbox(
        "Brand",
        sorted(product_df["brand"].unique())
    )

    min_budget = st.number_input(
        "Minimum Price (USD)",
        min_value=float(price_min),
        max_value=float(price_max),
        value=500.0,
        step=10.0
    )

    max_budget = st.number_input(
        "Maximum Price (USD)",
        min_value=float(price_min),
        max_value=float(price_max),
        value=1000.0,
        step=10.0
    )

    target_price = (min_budget + max_budget) / 2

    overall_rating = st.slider(
        "Overall Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0,
        step=0.1
    )

with col2:
    performance_rating = st.slider(
        "Performance",
        1.0, 5.0, 4.0, 0.1
    )

    camera_rating = st.slider(
        "Camera",
        1.0, 5.0, 4.0, 0.1
    )

    battery_rating = st.slider(
        "Battery Life",
        1.0, 5.0, 4.0, 0.1
    )

    display_rating = st.slider(
        "Display",
        1.0, 5.0, 4.0, 0.1
    )

    design_rating = st.slider(
        "Design",
        1.0, 5.0, 4.0, 0.1
    )

st.caption(
    f"Target price used for similarity calculation: ${target_price:,.2f}"
)

if st.button(
    "🔍 Get Recommendations",
    type="primary",
    use_container_width=True
):
    # Validate the price range before running the recommendation engine.
    if min_budget > max_budget:
        st.error(
            "Minimum Price cannot be greater than Maximum Price."
        )

    else:
        recommendations = get_recommendations(
            brand=selected_brand,
            price_usd=target_price,
            rating=overall_rating,
            performance_rating=performance_rating,
            camera_rating=camera_rating,
            battery_life_rating=battery_rating,
            display_rating=display_rating,
            design_rating=design_rating,
            top_n=10
        )

        st.divider()
        st.subheader("Recommended Mobile Products")

        display_df = recommendations.copy()
        display_df["price_usd"] = display_df["price_usd"].round(2)
        display_df["final_score"] = display_df["final_score"].round(4)

        display_df = display_df.rename(
            columns={
                "brand": "Brand",
                "model": "Model",
                "price_usd": "Price (USD)",
                "rating": "Rating",
                "performance_rating": "Performance",
                "camera_rating": "Camera",
                "battery_life_rating": "Battery",
                "display_rating": "Display",
                "design_rating": "Design",
                "final_score": "Recommendation Score"
            }
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"Top {len(display_df)} recommendations generated successfully."
        )

st.divider()

st.caption(
    "Recommendation method: weighted feature similarity + "
    "Cosine Similarity + brand-profile similarity."
)
