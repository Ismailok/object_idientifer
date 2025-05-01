import streamlit as st
from PIL import Image
import torch
import clip

# Charger le modèle CLIP
device = "cpu"  # Plus stable, évite les erreurs CUDA/cuDNN
model, preprocess = clip.load("ViT-B/32", device=device)

# Charger les labels depuis le fichier
with open("labels.txt", "r") as f:
    candidate_labels = [line.strip() for line in f.readlines()]

# Configuration de la page Streamlit
st.set_page_config(page_title="AI Object Identifier", layout="centered")
st.title("🖼️ AI Object Identifier")
st.markdown("Upload an image and let the AI identify **what object** is in it (cat, dog, etc.).")

# Upload de l’image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prétraitement
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in candidate_labels]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Obtenir la meilleure prédiction
    probs = similarity.squeeze().tolist()
    best_label = candidate_labels[probs.index(max(probs))]
    confidence = max(probs) * 100

    st.markdown("### 🧠 Prediction")
    st.success(f"**{best_label.capitalize()}** ({confidence:.2f}% confidence)")

# Section estimation des coûts
st.markdown("---")
st.markdown("## 🧾 Cost Estimate")
st.markdown("""
- **Model Used**: OpenAI CLIP (open-source via PyTorch)
- **Hosting Option**: Local CPU (no API cost)
- **Cost**: Free on local machine. Minimal cost on a cloud instance (e.g., EC2 t3.small ≈ $0.023/hr)
- **OpenAI Vision API Alternative**: ~$0.002/image (optional)

No API keys required. Fully local and free.
""")
