# 📱 Mobile Product Segmentation & Recommendation System

## 1. Project Overview

This project analyzes mobile customer review data to identify product
segments and provide personalized mobile product recommendations.

The project combines **Exploratory Data Analysis, K-Means Clustering,
feature similarity, Cosine Similarity, and a Streamlit application** to
help users discover mobile models based on their preferred brand,
budget, ratings, and product features.

------------------------------------------------------------------------

## 2. Product Segmentation

K-Means clustering is used to segment mobile products based on the
following features:

-   Price
-   Overall Rating
-   Battery Life Rating
-   Camera Rating
-   Performance Rating
-   Design Rating
-   Display Rating

Since these features operate on different scales, the numerical features
are standardized before applying K-Means.

### Cluster Selection

The **Elbow Method** and **Silhouette Score** were evaluated for
different values of K.

Although K=2 produced the highest Silhouette Score, **K=4 was selected
for this project because it provided more meaningful and interpretable
product segments for the business objective and recommendation
analysis.**

The four clusters show differences in:

-   Price levels
-   Overall ratings
-   Product specifications
-   Product quality characteristics

### Important Architecture Note

K-Means is used for **product segmentation and cluster analysis**.

It is **not directly used to calculate the recommendation score**.

------------------------------------------------------------------------

## 3. Recommendation System

The recommendation system operates at the **product/model level**.

Since the original dataset contains multiple customer reviews for the
same mobile model, reviews are aggregated by:

**Brand + Model**

Average values are calculated for the relevant product features to
create a product-level profile.

The recommendation system considers:

-   Price
-   Overall Rating
-   Performance
-   Camera
-   Battery Life
-   Display
-   Design
-   Preferred Brand

------------------------------------------------------------------------

## 4. Feature Normalization

The product features are normalized to a common **0--1 scale**.

For price:

``` text
Normalized Price =
(Price - Minimum Price) /
(Maximum Price - Minimum Price)
```

For ratings and product specifications:

``` text
Normalized Rating =
(Rating - 1) / 4
```

Feature similarity is then calculated as:

``` text
Feature Similarity =
1 - |Product Value - User Preference|
```

A value closer to **1** indicates that the product is closer to the
user's preference.

------------------------------------------------------------------------

## 5. Cosine Similarity

Cosine Similarity is used to compare the user's overall preference
vector with the normalized feature vector of each product.

It measures how closely the overall pattern of a product's attributes
matches the user's requested preferences.

A higher Cosine Similarity indicates greater overall similarity.

------------------------------------------------------------------------

## 6. Brand Similarity

The selected brand is treated as a **preference rather than a strict
filter**.

Brand profiles are created using the normalized product attributes of
each brand.

Cosine Similarity is then used to calculate the similarity between the
user's selected brand profile and other brand profiles.

This allows the recommendation system to consider:

-   Products from the preferred brand
-   Similar products from other brands

Therefore, selecting a brand does **not** automatically exclude other
brands.

------------------------------------------------------------------------

## 7. Recommendation Score

The final recommendation score combines individual feature similarity,
overall Cosine Similarity, and brand similarity.

  Component               Weight
  ------------------- ----------
  Price Similarity           20%
  Overall Rating             12%
  Performance                12%
  Camera                      8%
  Battery Life                8%
  Display                     8%
  Design                      4%
  Cosine Similarity          18%
  Brand Similarity           10%
  **Total**             **100%**

The final score is calculated as a weighted combination of these
components.

Products are then sorted in descending order of the final score.

A higher score indicates greater similarity to the user's selected
preferences.

------------------------------------------------------------------------

## 8. User Inputs

The Streamlit application allows the user to specify:

### Brand

The user's preferred mobile brand.

### Price Range

The user enters:

-   Minimum Price
-   Maximum Price

The midpoint of the selected range is used as the **target price for
price similarity**.

### Product Preferences

The user can specify desired ratings from **1 to 5** for:

-   Overall Rating
-   Performance
-   Camera
-   Battery Life
-   Display
-   Design

------------------------------------------------------------------------

## 9. Recommendation Output

The application returns the top recommended mobile models with:

-   Brand
-   Model
-   Price
-   Overall Rating
-   Performance
-   Camera
-   Battery
-   Display
-   Design
-   Recommendation Score

The recommendations are ranked from the highest to the lowest
recommendation score.

------------------------------------------------------------------------

## 10. Model Package

The recommendation model information is stored in:

``` text
Output/
└── mobile_recommendation_model.pkl
```

The package contains:

-   Product-level dataset
-   Normalized product features
-   Brand profiles
-   Minimum and maximum product prices
-   Product feature definitions
-   Rating feature definitions
-   Recommendation weights

The Streamlit application loads this package and uses it to generate
recommendations.

------------------------------------------------------------------------

## 11. Streamlit Application

The application is built using **Streamlit**.

The application:

1.  Loads the recommendation model package.
2.  Accepts customer preferences.
3.  Normalizes the selected preferences.
4.  Calculates feature-level similarity.
5.  Calculates Cosine Similarity.
6.  Calculates brand-profile similarity.
7.  Calculates the weighted recommendation score.
8.  Ranks the products.
9.  Displays the top recommendations.

------------------------------------------------------------------------

## 12. Recommendation Workflow

``` text
Customer Preferences
        ↓
Brand + Price + Ratings + Specifications
        ↓
Feature Normalization
        ↓
Feature Similarity
        ↓
Cosine Similarity
        ↓
Brand-Profile Similarity
        ↓
Weighted Recommendation Score
        ↓
Rank Products
        ↓
Top Mobile Model Recommendations
```

------------------------------------------------------------------------

## 13. Project Architecture

``` text
Mobile_Product_Segmentation/
│
├── .venv/
│
├── Input/
│
├── Output/
│   ├── Mobile_Reviews_Cleaned.csv
│   └── mobile_recommendation_model.pkl
│
├── data_Cleaning.ipynb
├── EDA.ipynb
├── Mobile_app.py
└── README.md
```

------------------------------------------------------------------------

## 14. Key Technologies

-   **Python**
-   **Pandas**
-   **NumPy**
-   **Matplotlib**
-   **Seaborn**
-   **Scikit-learn**
-   **K-Means Clustering**
-   **Elbow Method**
-   **Silhouette Score**
-   **Cosine Similarity**
-   **Streamlit**
-   **Pickle**

------------------------------------------------------------------------

## 15. Key Project Insights

The analysis helps identify:

-   Different product segments based on price and quality.
-   Relationships between product price and customer ratings.
-   Product quality patterns across different models.
-   Brand-level product characteristics.
-   Similar products based on customer preferences.
-   How price, ratings, specifications, and brand preference contribute
    to recommendations.

------------------------------------------------------------------------

## 16. Final Model Design

The project uses **two complementary machine-learning components**.

### Segmentation

**K-Means → Product Clusters**

Used to understand and segment the mobile product portfolio.

### Recommendation

**Weighted Feature Similarity + Cosine Similarity + Brand Similarity →
Product Recommendations**

Used to personalize recommendations for the customer.

This separation allows the project to use clustering for
**business/product segmentation** while using similarity-based methods
for **personalized recommendation**.
