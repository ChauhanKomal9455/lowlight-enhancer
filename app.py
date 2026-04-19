
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys, os, io
import gdown

# ── Auto Download Weights ────────────────────────────
WEIGHT_PATH = 'zerodce_best.pth'
FILE_ID     = '1fnunvtQSpgXQzpccv4MVook265lRo0wL'

@st.cache_resource
def load_model():
    if not os.path.exists(WEIGHT_PATH):
        with st.spinner("Downloading model weights..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            gdown.download(url, WEIGHT_PATH, quiet=False)

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models.zero_dce import ZeroDCE

    model = ZeroDCE()
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location='cpu'))
    model.eval()
    return model

# ── Enhance Function ─────────────────────────────────
def enhance_image(model, image: Image.Image) -> Image.Image:
    orig_size = image.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        enhanced = model(tensor).squeeze(0)
    enhanced = enhanced.clamp(0, 1).permute(1, 2, 0).numpy()
    enhanced = (enhanced * 255).astype(np.uint8)
    return Image.fromarray(enhanced).resize(orig_size)

def get_brightness(img):
    arr = np.array(img.convert('L'))
    return round(arr.mean(), 1)

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title = "Low-Light Image Enhancer",
    page_icon  = "🌙",
    layout     = "wide"
)

st.markdown("""
<style>
    .title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #1e2130;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffd200;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaaaaa;
    }
    .footer {
        text-align: center;
        color: #555;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #2e3250;
    }
    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid #2e3250;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────
st.markdown('<p class="title">Low-Light Image Enhancement</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Deep Learning Based Enhancement using Zero-DCE | B.Tech Project 2024-25</p>',
            unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## About This Project")
    st.markdown("""
    **Model:** Zero-DCE  
    **Dataset:** LOL Dataset (485 images)  
    **Training:** 100 Epochs  
    **Framework:** PyTorch  

    ---
    ### How it works
    Zero-DCE estimates pixel-wise curve parameters
    to enhance low-light images without needing
    paired training data.

    ---
    ### Metrics Achieved
    - **PSNR:** 16.33 dB
    - **SSIM:** 0.4600

    ---
    ### Project Info
    - B.Tech Project 2024-25
    - Dept. of CSE
    """)

# ── Main ─────────────────────────────────────────────
model = load_model()

uploaded = st.file_uploader(
    "Upload a Low-Light Image",
    type=["jpg", "jpeg", "png"],
    help="Upload any dark or low-light image"
)

if uploaded:
    image    = Image.open(uploaded).convert("RGB")

    with st.spinner("Enhancing your image..."):
        enhanced = enhance_image(model, image)

    orig_brightness = get_brightness(image)
    enh_brightness  = get_brightness(enhanced)
    improvement     = round(((enh_brightness - orig_brightness) / orig_brightness) * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{orig_brightness}</div><div class="metric-label">Original Brightness</div></div>',
                    unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{enh_brightness}</div><div class="metric-label">Enhanced Brightness</div></div>',
                    unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">+{improvement}%</div><div class="metric-label">Brightness Improvement</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.markdown("#### Enhanced Image")
        st.image(enhanced, use_column_width=True)

    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    st.download_button(
        label               = "Download Enhanced Image",
        data                = buf.getvalue(),
        file_name           = f"enhanced_{uploaded.name}",
        mime                = "image/png",
        use_container_width = True
    )

else:
    st.info("Upload a low-light image from the box above to get started!")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-box"><div class="metric-value">485</div><div class="metric-label">Training Images</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-box"><div class="metric-value">100</div><div class="metric-label">Epochs Trained</div></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-box"><div class="metric-value">79K</div><div class="metric-label">Model Parameters</div></div>',
                    unsafe_allow_html=True)

st.markdown('<div class="footer">Low-Light Image Enhancement using Deep Learning | B.Tech Project 2024-25</div>',
            unsafe_allow_html=True)
