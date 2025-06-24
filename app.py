
import streamlit as st
from PIL import Image, ImageDraw
import pandas as pd
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection, pipeline

# Initialize models
@st.cache_resource
def load_models():
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return processor, model, classifier

processor, model, classifier = load_models()

DEFECT_LABELS = [
    "particle contamination", "scratch",
    "pattern defect", "resist bubble", "etch anomaly"
]
ACTIONS = {
    "particle contamination": "Perform an IPA rinse and inspect gowning.",
    "scratch": "Check handling protocols to avoid abrasion.",
    "pattern defect": "Verify photomask alignment integrity.",
    "resist bubble": "Improve spin-coating parameters.",
    "etch anomaly": "Calibrate etch chemistry and run endpoint detection."
}

def detect_and_log(pil_img, conf_thresh=0.7, min_area=500.0):
    inputs = processor(images=pil_img, return_tensors="pt")
    outs = model(**inputs)
    target_sizes = torch.tensor([pil_img.size[::-1]])
    results = processor.post_process_object_detection(
        outs, target_sizes=target_sizes, threshold=conf_thresh
    )[0]

    raw = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        xmin, ymin, xmax, ymax = box.tolist()
        area = (xmax - xmin)*(ymax - ymin)
        if score < conf_thresh or area < min_area:
            continue
        raw.append({
            "label": model.config.id2label[label.item()],
            "score": float(score),
            "box": [float(xmin), float(ymin), float(xmax), float(ymax)]
        })

    entries = []
    for i, d in enumerate(raw, 1):
        txt = f"A region detected as '{d['label']}' at box {list(map(round, d['box']))}."
        defect = classifier(txt, DEFECT_LABELS)["labels"][0]
        coords = [round(x,1) for x in d['box']]
        action = ACTIONS[defect]
        entries.append({
            "defect_type": defect,
            "confidence": d['score'],
            "box": coords,
            "log": f"{i}. {defect.title()} at {coords} (conf {d['score']:.2f}); {action}"
        })
    df = pd.DataFrame(entries)
    log_text = "\n".join(df["log"].tolist())

    img_out = pil_img.copy()
    draw = ImageDraw.Draw(img_out)
    for _, d in df.iterrows():
        xmin, ymin, xmax, ymax = d["box"]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin-10),
                  f"{d['defect_type']} {d['confidence']:.2f}",
                  fill="red")
    return img_out, log_text, df

# Streamlit UI
st.title("Defect Detection and Classification App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Processing..."):
            img_out, log_text, df = detect_and_log(image)
        st.image(img_out, caption="Detected Defects", use_column_width=True)
        st.text_area("Detection Log", log_text, height=300)
        st.dataframe(df)
