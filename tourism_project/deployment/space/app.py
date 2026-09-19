"""
Gradio front end for the Wellness Tourism Package propensity model.

Deployed to a Hugging Face Gradio Space. Loads the registered model and its
metadata from the Hugging Face model hub and scores a single customer.
"""

import json
import os

import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# --- Fill in your Hugging Face username ---------------------------------
HF_USERNAME = "AhmedBenHamouda"
# ------------------------------------------------------------------------

MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"

model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_tourism_model_v1.joblib")
meta_path = hf_hub_download(repo_id=MODEL_REPO, filename="model_metadata.json")

model = joblib.load(model_path)
with open(meta_path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)

THRESHOLD = meta.get("decision_threshold", 0.5)
LEVELS = meta.get("category_levels", {})
FEATURE_ORDER = meta["feature_order"]
TUNED = meta.get("test_metrics_tuned", {})


def choices(column, fallback):
    """Offer exactly the category levels the model was trained on."""
    return LEVELS.get(column, fallback)


def predict(age, gender, marital_status, occupation, designation, monthly_income,
            city_tier, passport, own_car, number_of_trips, preferred_property_star,
            number_of_person_visiting, number_of_children_visiting,
            type_of_contact, duration_of_pitch, number_of_followups,
            product_pitched, pitch_satisfaction_score):
    """Score one customer and return the probability plus the recommended action."""
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
    probability = float(model.predict_proba(frame)[0, 1])

    label = {"Will purchase": probability, "Will not purchase": 1 - probability}

    if probability >= THRESHOLD:
        verdict = (
            f"### High propensity\n"
            f"Purchase probability **{probability * 100:.1f}%**, above the "
            f"{THRESHOLD * 100:.1f}% decision threshold.\n\n"
            f"**Recommended action:** prioritise for contact and plan up to five follow ups."
        )
    else:
        verdict = (
            f"### Low propensity\n"
            f"Purchase probability **{probability * 100:.1f}%**, below the "
            f"{THRESHOLD * 100:.1f}% decision threshold.\n\n"
            f"**Recommended action:** deprioritise, or place in a low cost nurture campaign."
        )
    verdict += "\n\nThe base conversion rate across the customer base is 19.3%."
    return label, verdict


with gr.Blocks(title="Wellness Package Propensity") as demo:
    gr.Markdown("# Wellness Tourism Package: Purchase Propensity")
    gr.Markdown(
        "Enter a customer profile to estimate the probability that they will purchase "
        "the Wellness Tourism Package, and see the recommended sales action."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("**Demographics**")
            age = gr.Number(label="Age", value=35, minimum=18, maximum=75)
            gender = gr.Dropdown(choices("Gender", ["Male", "Female"]),
                                 label="Gender", value="Male")
            marital_status = gr.Dropdown(choices("MaritalStatus", ["Single", "Married", "Divorced"]),
                                         label="Marital status", value="Single")
            occupation = gr.Dropdown(choices("Occupation", ["Salaried", "Small Business",
                                                            "Large Business", "Free Lancer"]),
                                     label="Occupation", value="Salaried")
            designation = gr.Dropdown(choices("Designation", ["Executive", "Manager",
                                                              "Senior Manager", "AVP", "VP"]),
                                      label="Designation", value="Executive")
            monthly_income = gr.Number(label="Monthly income", value=22000,
                                       minimum=1000, maximum=100000)

        with gr.Column():
            gr.Markdown("**Travel profile**")
            city_tier = gr.Dropdown([1, 2, 3], label="City tier", value=1)
            passport = gr.Radio(["Yes", "No"], label="Holds a valid passport", value="No")
            own_car = gr.Radio(["Yes", "No"], label="Owns a car", value="Yes")
            number_of_trips = gr.Number(label="Average trips per year", value=3,
                                        minimum=0, maximum=25)
            preferred_property_star = gr.Dropdown([3.0, 4.0, 5.0],
                                                  label="Preferred property rating", value=3.0)
            number_of_person_visiting = gr.Number(label="People travelling", value=3,
                                                  minimum=1, maximum=6)
            number_of_children_visiting = gr.Number(label="Children under 5 travelling",
                                                    value=1, minimum=0, maximum=5)

        with gr.Column():
            gr.Markdown("**Interaction history**")
            gr.Markdown("_Leave at the defaults when scoring a customer who has not been "
                        "contacted yet._")
            type_of_contact = gr.Dropdown(choices("TypeofContact", ["Self Enquiry", "Company Invited"]),
                                          label="Type of contact", value="Self Enquiry")
            duration_of_pitch = gr.Number(label="Duration of pitch (minutes)", value=15,
                                          minimum=0, maximum=60)
            number_of_followups = gr.Number(label="Number of follow ups", value=4,
                                            minimum=0, maximum=6)
            product_pitched = gr.Dropdown(choices("ProductPitched", ["Basic", "Deluxe", "Standard",
                                                                     "Super Deluxe", "King"]),
                                          label="Product pitched", value="Basic")
            pitch_satisfaction_score = gr.Dropdown([1, 2, 3, 4, 5],
                                                   label="Pitch satisfaction score", value=3)

    button = gr.Button("Predict purchase propensity", variant="primary")
    with gr.Row():
        output_label = gr.Label(label="Prediction", num_top_classes=2)
        output_text = gr.Markdown()

    button.click(
        predict,
        inputs=[age, gender, marital_status, occupation, designation, monthly_income,
                city_tier, passport, own_car, number_of_trips, preferred_property_star,
                number_of_person_visiting, number_of_children_visiting,
                type_of_contact, duration_of_pitch, number_of_followups,
                product_pitched, pitch_satisfaction_score],
        outputs=[output_label, output_text],
    )

    with gr.Accordion("How this model works", open=False):
        gr.Markdown(
            f"""
The model is a gradient boosted tree ensemble ({meta.get('model_family', 'XGBoost')}) trained on
4,128 historical customer records. It was selected against Logistic Regression and Random Forest
using five fold cross validated F1 on the positive class, which balances the cost of missing a
buyer against the cost of a wasted sales call.

Held out test performance at the tuned threshold: F1 {TUNED.get('f1', 0):.3f},
recall {TUNED.get('recall', 0):.3f}, ROC AUC {TUNED.get('roc_auc', 0):.3f}.

The strongest drivers of purchase are passport ownership, job grade (Executives convert far
better than senior management), marital status and city tier. The decision threshold of
{THRESHOLD:.3f} was tuned on out of fold predictions, not left at the default 0.5.
            """
        )

if __name__ == "__main__":
    demo.launch()
