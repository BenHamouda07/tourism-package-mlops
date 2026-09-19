"""
Streamlit front end for the Wellness Tourism Package propensity model.

Loads the registered model and its metadata from the Hugging Face model hub and
scores a single customer, returning the purchase probability and the recommended
action at the operating threshold chosen during training.
"""

import json

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "YOUR_HF_USERNAME"
# ------------------------------------------------------------------------

MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

st.set_page_config(page_title="Wellness Package Propensity", layout="wide")


@st.cache_resource
def load_artifacts():
    """Download the registered model and metadata once per container start."""
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_tourism_model_v1.joblib")
    meta_path = hf_hub_download(repo_id=MODEL_REPO, filename="model_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return joblib.load(model_path), metadata


model, meta = load_artifacts()
THRESHOLD = meta.get("decision_threshold", 0.5)
LEVELS = meta.get("category_levels", {})
FEATURE_ORDER = meta["feature_order"]

st.title("Wellness Tourism Package: Purchase Propensity")
st.write(
    "Enter a customer profile to estimate the probability that they will purchase "
    "the Wellness Tourism Package, and see the recommended sales action."
)

with st.sidebar:
    st.header("Model in production")
    st.write(f"**Family:** {meta.get('model_family', 'n/a')}")
    st.write(f"**Decision threshold:** {THRESHOLD:.3f}")
    tuned = meta.get("test_metrics_tuned", {})
    if tuned:
        st.write(f"**Test F1:** {tuned.get('f1', 0):.3f}")
        st.write(f"**Test recall:** {tuned.get('recall', 0):.3f}")
        st.write(f"**Test ROC AUC:** {tuned.get('roc_auc', 0):.3f}")
    st.caption(f"Loaded from {MODEL_REPO}")


def choices(column, fallback):
    """Offer exactly the category levels the model was trained on."""
    return LEVELS.get(column, fallback)


st.subheader("Customer profile")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    age = st.number_input("Age", min_value=18, max_value=75, value=35, step=1)
    gender = st.selectbox("Gender", choices("Gender", ["Male", "Female"]))
    marital_status = st.selectbox("Marital status",
                                  choices("MaritalStatus", ["Single", "Married", "Divorced"]))
    occupation = st.selectbox("Occupation",
                              choices("Occupation", ["Salaried", "Small Business",
                                                     "Large Business", "Free Lancer"]))
    designation = st.selectbox("Designation",
                               choices("Designation", ["Executive", "Manager",
                                                       "Senior Manager", "AVP", "VP"]))
    monthly_income = st.number_input("Monthly income", min_value=1000, max_value=100000,
                                     value=22000, step=500)

with col2:
    st.markdown("**Travel profile**")
    city_tier = st.selectbox("City tier", [1, 2, 3], index=0,
                             help="Tier 1 is the most developed")
    passport = st.selectbox("Holds a valid passport", ["Yes", "No"], index=1)
    own_car = st.selectbox("Owns a car", ["Yes", "No"], index=0)
    number_of_trips = st.number_input("Average trips per year", min_value=0, max_value=25,
                                      value=3, step=1)
    preferred_property_star = st.selectbox("Preferred property rating", [3.0, 4.0, 5.0], index=0)
    number_of_person_visiting = st.number_input("People travelling", min_value=1, max_value=6,
                                                value=3, step=1)
    number_of_children_visiting = st.number_input("Children under 5 travelling",
                                                  min_value=0, max_value=5, value=1, step=1)

with col3:
    st.markdown("**Interaction history**")
    st.caption("Leave at the defaults when scoring a customer who has not been contacted yet.")
    type_of_contact = st.selectbox("Type of contact",
                                   choices("TypeofContact", ["Self Enquiry", "Company Invited"]))
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=60,
                                        value=15, step=1)
    number_of_followups = st.number_input("Number of follow ups", min_value=0, max_value=6,
                                          value=4, step=1)
    product_pitched = st.selectbox("Product pitched",
                                   choices("ProductPitched", ["Basic", "Deluxe", "Standard",
                                                              "Super Deluxe", "King"]))
    pitch_satisfaction_score = st.selectbox("Pitch satisfaction score", [1, 2, 3, 4, 5], index=2)

record = {
    "Age": float(age),
    "TypeofContact": type_of_contact,
    "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(number_of_person_visiting),
    "NumberOfFollowups": float(number_of_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_property_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(number_of_trips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitch_satisfaction_score),
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": float(number_of_children_visiting),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income),
}

# Reorder the columns to match the training schema exactly.
frame = pd.DataFrame([record])[FEATURE_ORDER]

if st.button("Predict purchase propensity", type="primary"):
    probability = float(model.predict_proba(frame)[0, 1])
    will_buy = probability >= THRESHOLD

    st.subheader("Result")
    left, right = st.columns(2)
    left.metric("Purchase probability", f"{probability * 100:.1f}%")
    right.metric("Decision threshold", f"{THRESHOLD * 100:.1f}%")
    st.progress(min(probability, 1.0))

    if will_buy:
        st.success(
            f"**High propensity.** This customer is predicted to purchase the Wellness Package "
            f"({probability * 100:.1f}% probability, above the {THRESHOLD * 100:.1f}% threshold). "
            "Recommended action: prioritise for contact and plan up to five follow ups."
        )
    else:
        st.warning(
            f"**Low propensity.** This customer is not predicted to purchase "
            f"({probability * 100:.1f}% probability, below the {THRESHOLD * 100:.1f}% threshold). "
            "Recommended action: deprioritise, or place in a low cost nurture campaign."
        )

    st.caption(
        "The base conversion rate across the customer base is 19.3%. "
        "Any probability well above that represents a stronger than average prospect."
    )

with st.expander("How this model works"):
    st.markdown(
        """
The model is a gradient boosted tree ensemble trained on 4,128 historical customer records.
It was selected against Logistic Regression and Random Forest using five fold cross validated
F1 on the positive class, which balances the cost of missing a buyer against the cost of a
wasted sales call.

The strongest drivers of purchase are passport ownership, job grade (Executives convert far
better than senior management), marital status and city tier. The decision threshold shown in
the sidebar was tuned on out of fold predictions to maximise F1, not left at the default 0.5.
        """
    )
