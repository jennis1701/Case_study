"""
app.py
------
Streamlit front-end for the Explainable Bloom's Taxonomy Classifier.

UI-only logic lives here.  All model/inference/parsing code is in src/.
        if st.session_state.pop("ground_truth_dataset_hash", None) is not None:
            st.session_state.pop("matched_dataset_evaluations", None)
            st.session_state.pop("last_result", None)

Run with:
    streamlit run app.py
"""

import hashlib
import logging
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Streamlit page configuration — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Bloom's Taxonomy Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Imports (after page config)
# ---------------------------------------------------------------------------
from config import (
    BLOOM_LEVELS,
    BLOOM_DEFINITIONS,
    MODEL_NAME,
    RESEARCH_DISCLAIMER,
)
from src.model_service import ModelService
from src.bloom_classifier import BloomClassifier
from src.evaluation import (
    compute_metrics,
    find_ground_truth,
    get_bloom_dataset_example,
    validate_bloom_dataset,
)
from src.utils import (
    sanitise_text_input,
    log_inference_result,
    get_bloom_color,
    get_bloom_icon,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom CSS — light academic theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #F7F9FC;
    color: #1C2B3A;
    font-family: 'Segoe UI', 'Inter', sans-serif;
}

/* ---- Header area ---- */
.app-header {
    text-align: center;
    padding: 2rem 1rem 1rem 1rem;
    border-bottom: 2px solid #D6E4F0;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-size: 2.1rem;
    font-weight: 700;
    color: #1A3A5C;
    margin: 0;
}
.app-header p {
    font-size: 1.05rem;
    color: #4A6785;
    margin: 0.4rem 0 0 0;
}

/* ---- Section headings ---- */
.section-heading {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1A3A5C;
    border-left: 4px solid #2E86C1;
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* ---- Result card ---- */
.result-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 1.6rem 2rem;
    box-shadow: 0 2px 12px rgba(30,60,100,0.08);
    border-top: 5px solid #2E86C1;
    margin-bottom: 1.2rem;
}

/* ---- Bloom level badge ---- */
.bloom-badge {
    display: inline-block;
    font-size: 1.6rem;
    font-weight: 700;
    color: #FFFFFF;
    background: linear-gradient(135deg, #1A3A5C, #2E86C1);
    border-radius: 8px;
    padding: 0.4rem 1.4rem;
    letter-spacing: 0.05em;
}

/* ---- Taxonomy reference cards ---- */
.taxonomy-card {
    background: #FFFFFF;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    border-left: 4px solid #2E86C1;
    box-shadow: 0 1px 5px rgba(30,60,100,0.07);
    height: 100%;
}
.taxonomy-card .level-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1A3A5C;
}
.taxonomy-card .level-def {
    font-size: 0.82rem;
    color: #5A728A;
    margin-top: 0.25rem;
    line-height: 1.45;
}

/* ---- Disclaimer ---- */
.disclaimer {
    background: #EBF5FB;
    border: 1px solid #AED6F1;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #1A5276;
    margin-top: 1rem;
}

/* ---- Evidence/Explanation text ---- */
.evidence-box {
    background: #F0F7FF;
    border-left: 4px solid #5DADE2;
    border-radius: 4px;
    padding: 0.8rem 1rem;
    font-style: italic;
    color: #1C3A5C;
    font-size: 0.92rem;
}

/* ---- Confidence bar label ---- */
.conf-label {
    font-size: 0.78rem;
    color: #7F8C8D;
    margin-top: 0.15rem;
}

/* ---- Streamlit element overrides ---- */
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #C8D8E8 !important;
    border-radius: 6px !important;
}

.stButton button {
    background: linear-gradient(135deg, #1A3A5C, #2E86C1) !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.55rem 2rem !important;
}
.stButton button:hover {
    opacity: 0.9 !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Model loading — cached so it runs ONCE per Streamlit session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading Gemma 3 model (this may take a few minutes)...")
def load_model_service() -> ModelService:
    """
    Load the ModelService exactly once.
    The result is cached by st.cache_resource and reused across reruns.
    """
    return ModelService(temperature=0.0, max_new_tokens=8)


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="app-header">
            <h1>🎓 Explainable Bloom's Taxonomy Classifier</h1>
            <p>LLM-Based Classification of Student Responses &nbsp;|&nbsp; Powered by Gemma 3</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------

def render_input_section() -> tuple[str, str, str]:
    """Render inputs and return question, response, and subject."""
    st.markdown('<div class="section-heading">📝 Input</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 1], gap="large")

    with col_left:
        question = st.text_area(
            "Question",
            placeholder="Enter the question posed to the student...",
            height=120,
            key="input_question",
        )

        student_response = st.text_area(
            "Student Response",
            placeholder=(
                "Enter the student's free-form answer here. "
                "The system will classify the cognitive level demonstrated."
            ),
            height=180,
            key="input_response",
        )

    with col_right:
        subject = st.text_input(
            "Subject / Domain (optional)",
            placeholder="e.g., Computer Science",
            key="input_subject",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<small style='color:#5A728A;'>Providing the subject domain helps "
            "the model contextualise the response, but it is not required.</small>",
            unsafe_allow_html=True,
        )

    return question, student_response, subject


# ---------------------------------------------------------------------------
# Result section
# ---------------------------------------------------------------------------

def render_result(
    result,
    ground_truth_level: str | None,
    dataset_loaded: bool,
    ground_truth_source: str | None,
) -> None:
    """Render one prediction and scores for matched CSV rows this session."""
    st.markdown(
        '<div class="section-heading">📊 Classification Result</div>',
        unsafe_allow_html=True,
    )

    if ground_truth_level is not None:
        reference_name = (
            "CSV ground truth" if ground_truth_source == "csv" else "Manual reference"
        )
        if result.is_valid and result.bloom_level == ground_truth_level:
            st.success(f"{reference_name}: {ground_truth_level}. Prediction matches.")
        elif result.is_valid:
            st.warning(
                f"{reference_name}: {ground_truth_level}. "
                f"Gemma predicted {result.bloom_level}."
            )
        else:
            st.warning(
                f"{reference_name}: {ground_truth_level}. "
                "The model output was invalid and counts as incorrect."
            )
    elif dataset_loaded:
        st.info(
            "No matching question and response were found in the CSV; "
            "load a row from the CSV above to include a prediction in the scores."
        )
    else:
        st.info("Upload a labeled CSV to compare this prediction with ground truth.")

    if result.is_valid:
        color = get_bloom_color(result.bloom_level)
        icon = get_bloom_icon(result.bloom_level)
        st.markdown(
            f"""
            <div class="result-card">
                <div style="margin-bottom:0.5rem; font-size:0.85rem; color:#7F8C8D; font-weight:600; text-transform:uppercase;">
                    Predicted Bloom Level
                </div>
                <div class="bloom-badge" style="background: linear-gradient(135deg, #1A3A5C, {color});">
                    {icon}&nbsp; {result.bloom_level}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(f"Parsed Prediction: {result.bloom_level}")
    else:
        st.error("Invalid model output — prediction not available")
        st.caption(
            f"Detail: {result.parse_error or 'The model did not return a valid Bloom label.'}"
        )

    history = list(st.session_state.get("matched_dataset_evaluations", {}).values())
    if history:
        metrics = compute_metrics(
            [item["ground_truth"] for item in history],
            [item["predicted_bloom_level"] for item in history],
        )
        st.markdown(f"**Session scores over {len(history)} matched dataset row(s)**")
        metric_columns = st.columns(4)
        metric_columns[0].metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
        metric_columns[1].metric("Macro Precision", f"{metrics['macro_precision'] * 100:.2f}%")
        metric_columns[2].metric("Macro Recall", f"{metrics['macro_recall'] * 100:.2f}%")
        metric_columns[3].metric("Macro F1-score", f"{metrics['macro_f1'] * 100:.2f}%")
        valid_count = sum(item["predicted_bloom_level"] != "Invalid" for item in history)
        st.caption(
            f"Matched rows scored: {len(history)} | "
            f"Valid prediction coverage: {valid_count / len(history) * 100:.2f}% | "
            "Invalid outputs count as incorrect."
        )
        with st.expander("Session confusion matrix and scored rows"):
            matrix = pd.DataFrame(
                metrics["confusion_matrix"],
                index=pd.Index(metrics["confusion_matrix_labels"], name="Actual"),
                columns=pd.Index(
                    metrics["confusion_matrix_prediction_labels"],
                    name="Predicted",
                ),
            )
            st.dataframe(matrix, use_container_width=True)
            st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.caption("Load and analyze a row from the CSV to build session scores.")

    if result.is_valid and result.explanation:
        st.markdown("**Explanation**")
        st.info(result.explanation)
    if result.is_valid and result.alternative_level and result.alternative_reason:
        with st.expander(f"Why not {result.alternative_level}?"):
            st.write(result.alternative_reason)

    with st.expander("🔬 Raw Gemma response (for research inspection)"):
        st.code(result.raw_output, language="text")

    # --- Disclaimer ---
    st.markdown(
        f'<div class="disclaimer">⚠️ <strong>Research Disclaimer:</strong> {RESEARCH_DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Ground-truth dataset
# ---------------------------------------------------------------------------

def render_ground_truth_dataset() -> pd.DataFrame | None:
    """Load and validate the labeled CSV used to score manual predictions."""
    st.markdown('<div class="section-heading">📊 Ground-Truth Dataset</div>', unsafe_allow_html=True)
    st.caption("Required columns: question, student_response, bloom_level")
    st.caption(
        "The CSV label is only used for scoring. Gemma receives the question and response."
    )
    uploaded_file = st.file_uploader(
        "Bloom reference dataset",
        type="csv",
        key="evaluation_dataset",
    )
    if uploaded_file is None:
        if st.session_state.pop("ground_truth_dataset_hash", None) is not None:
            st.session_state.pop("matched_dataset_evaluations", None)
            st.session_state.pop("last_result", None)
            st.session_state.pop("reference_row_selector", None)
        return

    dataset_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if dataset_hash != st.session_state.get("ground_truth_dataset_hash"):
        st.session_state["ground_truth_dataset_hash"] = dataset_hash
        st.session_state["matched_dataset_evaluations"] = {}
        st.session_state.pop("last_result", None)
        st.session_state.pop("reference_row_selector", None)

    try:
        dataset = pd.read_csv(uploaded_file)
    except (OSError, UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        st.error(f"The CSV could not be read: {exc}")
        return

    try:
        validate_bloom_dataset(dataset)
    except ValueError as exc:
        st.error(str(exc))
        return None
    if dataset.empty:
        st.error("The reference CSV has no rows.")
        return None

    st.caption(f"Loaded ground-truth rows: {len(dataset)}")
    examples = [
        get_bloom_dataset_example(dataset, row_position)
        for row_position in range(len(dataset))
    ]
    selected_position = st.selectbox(
        "Select a CSV question and response",
        options=range(len(examples)),
        format_func=lambda index: (
            f"{index + 1}. {examples[index][0][:70]} | "
            f"{examples[index][1][:70]} | {examples[index][2]}"
        ),
        key="reference_row_selector",
    )
    if st.button("Load selected row into the form", key="load_reference_row"):
        question, student_response, _ = examples[selected_position]
        st.session_state["input_question"] = question
        st.session_state["input_response"] = student_response
        st.rerun()

    return dataset


def _record_matched_evaluation(
    question: str,
    student_response: str,
    ground_truth: str,
    result,
) -> None:
    key = (
        " ".join(question.split()).casefold(),
        " ".join(student_response.split()).casefold(),
    )
    history = st.session_state.setdefault("matched_dataset_evaluations", {})
    prediction = result.bloom_level if result.is_valid else "Invalid"
    history[key] = {
        "question": question,
        "student_response": student_response,
        "ground_truth": ground_truth,
        "predicted_bloom_level": prediction,
        "is_correct": result.is_valid and prediction == ground_truth,
    }


# ---------------------------------------------------------------------------
# Bloom Taxonomy reference section
# ---------------------------------------------------------------------------

def render_taxonomy_reference() -> None:
    """Render the six Bloom level cards at the bottom of the page."""
    st.markdown("<hr style='border:1px solid #D6E4F0; margin: 2rem 0;'>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-heading">📚 Revised Bloom\'s Taxonomy Reference</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    for col, level in zip(cols, BLOOM_LEVELS):
        color = get_bloom_color(level)
        icon = get_bloom_icon(level)
        defn = BLOOM_DEFINITIONS[level]
        with col:
            st.markdown(
                f"""
                <div class="taxonomy-card" style="border-left-color:{color};">
                    <div class="level-name">{icon} {level}</div>
                    <div class="level-def">{defn}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar — model information
# ---------------------------------------------------------------------------

def render_sidebar(model_service: ModelService) -> None:
    with st.sidebar:
        st.markdown("### ⚙️ Model Configuration")
        st.markdown(f"**Model:** `{MODEL_NAME}`")

        if model_service.is_ready:
            st.success("Model loaded ✓")
        else:
            st.error(f"Model failed to load:\n\n{model_service.load_error}")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This application is part of an M.Tech AI & Data Science thesis research project. "
            "It classifies student responses into Revised Bloom's Taxonomy cognitive levels "
            "using Gemma 3 and structured prompt engineering."
        )
        st.markdown("---")
        st.markdown(
            "<small style='color:#7F8C8D;'>Model-reported confidence is not a "
            "statistically calibrated probability.</small>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    render_header()

    # Load model (cached)
    model_service = load_model_service()

    render_sidebar(model_service)

    ground_truth_dataset = render_ground_truth_dataset()
    st.markdown("---")

    # Input section
    question, student_response, subject = render_input_section()

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyse Response", use_container_width=False)

    if analyze_clicked:
        # --- Validate inputs ---
        q_clean = sanitise_text_input(question)
        r_clean = sanitise_text_input(student_response)

        if not q_clean:
            st.error("Please enter a question before analysing.")
            return
        if not r_clean:
            st.error("Please enter the student response before analysing.")
            return

        ground_truth_level = None
        ground_truth_source = None
        if ground_truth_dataset is not None:
            try:
                ground_truth_level = find_ground_truth(
                    ground_truth_dataset,
                    q_clean,
                    r_clean,
                )
            except ValueError as exc:
                st.error(str(exc))
                return
        if ground_truth_level is not None:
            ground_truth_source = "csv"

        # --- Check model is ready ---
        if not model_service.is_ready:
            st.error(
                "The model is not available. Please check the sidebar for details.\n\n"
                f"Error: {model_service.load_error}"
            )
            return

        # --- Run classification ---
        classifier = BloomClassifier(model_service)
        subject_clean = sanitise_text_input(subject, max_length=256)

        with st.spinner("Analysing the student response using Gemma 3..."):
            result = classifier.classify_evaluation(
                question=q_clean,
                student_response=r_clean,
                subject=subject_clean,
            )

        if ground_truth_source == "csv":
            _record_matched_evaluation(
                q_clean,
                r_clean,
                ground_truth_level,
                result,
            )
        st.session_state["last_result"] = (
            result,
            ground_truth_level,
            ground_truth_dataset is not None,
            ground_truth_source,
        )

        # --- Render result ---
        render_result(
            result,
            ground_truth_level,
            dataset_loaded=ground_truth_dataset is not None,
            ground_truth_source=ground_truth_source,
        )

        if result.is_valid:
            # --- Log result ---
            log_inference_result(
                question=q_clean,
                student_response=r_clean,
                predicted_level=result.bloom_level,
                confidence=result.confidence,
                explanation=result.explanation,
                subject=subject_clean,
                parse_error=result.parse_error,
            )

    elif st.session_state.get("last_result"):
        result, ground_truth_level, dataset_loaded, ground_truth_source = (
            st.session_state["last_result"]
        )
        render_result(
            result,
            ground_truth_level,
            dataset_loaded,
            ground_truth_source,
        )

    # Taxonomy reference always visible
    render_taxonomy_reference()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
