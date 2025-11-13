import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("C:\\Users\\subha\\OneDrive\\Desktop\\ML_Project\\model.pkl")
scaler = joblib.load("C:\\Users\\subha\\OneDrive\\Desktop\\ML_Project\\scaler.pkl")
encoders = joblib.load("C:\\Users\\subha\\OneDrive\\Desktop\\ML_Project\\encoders.pkl")
df = pd.read_csv("C:\\Users\\subha\\OneDrive\\Desktop\\ML_Project\\Ai_Health_Products.csv")



st.set_page_config(page_title="AI Health Product Analyzer", layout="wide")
st.title("🏋️‍♂️ AI Health Product Analyzer")
st.caption("Analyze health products, predict if it's beneficial, and get AI-powered recommendations.")

st.markdown("---")


st.subheader("Product Information")

col1, col2, col3 = st.columns(3)

with col1:
    Brand = st.selectbox("Brand", df["Brand"].unique())
    Category = st.selectbox("Category", df["Category"].unique())
    Country = st.selectbox("Country", df["Country"].unique())

with col2:
    Product_Type = st.selectbox("Product Type", df["Product_Type"].unique())
    Source_Type = st.selectbox("Source Type", df["Source_Type"].unique())
    Price = st.number_input("Price (INR)", 100, 200000, 2500)

with col3:
    Rating = st.slider("Average Rating", 1.0, 5.0, 4.2)
    Reviews = st.number_input("Number of Reviews", 0, 50000, 1200)
    Monthly_Sales = st.number_input("Monthly Sales", 0, 200000, 3000)

st.markdown("---")
st.subheader("Product Quality Scores")

col4, col5, col6 = st.columns(3)

with col4:
    IQ = st.slider("Ingredient Quality (%)", 30, 100, 85)
    Clin = st.slider("Clinical Approval (%)", 20, 100, 70)

with col5:
    Eco = st.slider("Eco-Friendliness (%)", 20, 100, 65)
    Trust = st.slider("User Trust Score (%)", 20, 100, 75)

with col6:
    Eff = st.slider("Effectiveness Score (%)", 20, 100, 80)
    Side = st.slider("Side Effect Rate (%)", 0.0, 30.0, 5.0)

User_HIS = st.slider("Your Expected Minimum Health Impact Score", 0, 100, 70)


input_dict = {
    "Brand": Brand,
    "Product_Type": Product_Type,
    "Category": Category,
    "Price_INR": Price,
    "Avg_Rating": Rating,
    "Num_Reviews": Reviews,
    "Ingredient_Quality(%)": IQ,
    "Clinical_Approval(%)": Clin,
    "Eco_Friendliness(%)": Eco,
    "User_Trust_Score(%)": Trust,
    "Effectiveness_Score(%)": Eff,
    "Side_Effect_Rate(%)": Side,
    "Monthly_Sales": Monthly_Sales,
    "Country": Country,
    "Source_Type": Source_Type,
    "Health_Score_User": User_HIS
}


encoded = input_dict.copy()
for col, encoder in encoders.items():
    try:
        encoded[col] = encoder.transform([encoded[col]])[0]
    except:
        encoded[col] = 0

feature_cols = [
    "Brand","Product_Type","Category","Price_INR","Avg_Rating","Num_Reviews",
    "Ingredient_Quality(%)","Clinical_Approval(%)","Eco_Friendliness(%)",
    "User_Trust_Score(%)","Effectiveness_Score(%)","Side_Effect_Rate(%)",
    "Monthly_Sales","Country","Source_Type"
]

X = pd.DataFrame([encoded], columns=feature_cols)
X_scaled = scaler.transform(X)


if st.button("Analyze Product"):
    pred = model.predict(X_scaled)[0]
    pred_label = "BENEFICIAL" if pred == 1 else "NOT BENEFICIAL"

    # Display result
    st.markdown("### Prediction Result")
    if pred == 1:
        st.success(f"✔ YES! This product is **{pred_label}**")
    else:
        st.error(f"✘ This product is **{pred_label}**")

    st.markdown("---")
    st.subheader("Top 5 AI-Recommended Healthier Alternatives")

    reco_cols = [
        "Price_INR","Avg_Rating","Ingredient_Quality(%)","Clinical_Approval(%)",
        "Eco_Friendliness(%)","User_Trust_Score(%)","Effectiveness_Score(%)",
        "Side_Effect_Rate(%)"
    ]

    reco_df = df[reco_cols]
    reco_norm = (reco_df - reco_df.min()) / (reco_df.max() - reco_df.min() + 1e-9)

    user_vec = np.array([
        Price, Rating, IQ, Clin, Eco, Trust, Eff, Side
    ]).reshape(1, -1)

    user_norm = (user_vec - reco_df.min().values) / (reco_df.max().values - reco_df.min().values + 1e-9)
    sims = np.dot(reco_norm.values, user_norm.T).flatten()

    df["similarity"] = sims

    rec = (
        df[df["Health_Impact_Score"] > User_HIS]
        .sort_values("similarity", ascending=False)
        .head(5)[["Product_ID","Product_Type","Health_Impact_Score","Avg_Rating"]]
        .reset_index(drop=True)
    )


    for i, row in rec.iterrows():
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; border:1px solid #ddd; margin-bottom:10px;">
                <h4>#{i+1} — {row['Product_Type']}</h4>
                <b>Product ID:</b> {row['Product_ID']}<br>
                <b>Health Impact Score:</b> {row['Health_Impact_Score']}<br>
                <b>Average Rating:</b> ⭐ {row['Avg_Rating']}
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.success("✔ Recommendations generated successfully!")
