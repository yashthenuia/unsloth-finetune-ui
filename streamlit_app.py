import streamlit as st

st.set_page_config(
    page_title="Unsloth Fine-Tuner",
    page_icon="🦥",
    layout="centered"
)

st.title("🦥 Unsloth Fine-Tuner")
st.caption("Fine-tune small LLMs for FREE using Google Colab (T4 GPU)")

st.divider()

# -----------------------------
# User inputs
# -----------------------------
model_name = st.selectbox(
    "Choose Base Model",
    [
        "unsloth/Phi-3-mini-4k-instruct",
        "unsloth/TinyLlama-1.1B",
        "unsloth/gemma-2b-it"
    ]
)

epochs = st.slider("Epochs", 1, 5, 1)
batch_size = st.selectbox("Batch Size", [1, 2, 4])
learning_rate = st.selectbox("Learning Rate", [2e-4, 1e-4, 5e-5])

st.divider()

# -----------------------------
# Open Colab
# -----------------------------
st.subheader("🚀 Step 1: Open Google Colab")

st.markdown(
    """
[👉 Open Colab Notebook (Free T4 GPU)](
https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/unsloth-finetune-ui/blob/main/colab/runner.ipynb
)
"""
)

# -----------------------------
# Command
# -----------------------------
st.subheader("▶ Step 2: Run ONE command in Colab")

command = f"""
!python train.py \\
  --model "{model_name}" \\
  --epochs {epochs} \\
  --batch_size {batch_size} \\
  --lr {learning_rate}
"""

st.code(command, language="bash")

st.success(
    "Upload your dataset in Colab → Training starts → Download merged model ZIP"
)
