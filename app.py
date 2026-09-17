import streamlit as st

from src.predict import predict_review


st.set_page_config(page_title="Review Toxicity Detector", page_icon="Shield")
st.title("Customer Review Toxicity Detector")
st.write("Check whether a customer review contains toxic language.")

review = st.text_area(
    "Customer review",
    placeholder="This product is terrible and the seller is an idiot.",
    height=140,
)

if st.button("Analyze review", type="primary"):
    try:
        result = predict_review(review)
    except (ValueError, FileNotFoundError) as error:
        st.error(str(error))
    else:
        if result["label"] == "Toxic":
            st.error(result["label"])
        else:
            st.success(result["label"])
        st.metric("Toxicity probability", f"{result['toxicity_probability']:.1%}")
        st.subheader("Category probabilities")
        for category, probability in result["categories"].items():
            st.write(f"**{category.title()}**: {probability:.1%}")
            st.progress(probability)