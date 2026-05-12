# Streamlit Cloud Deployment Guide

## Setup Instructions

### Option 1: Deploy Streamlit App to Streamlit Cloud (Recommended)

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in with GitHub**
3. **Click "Deploy an app"**
4. **Fill in the details:**
   - GitHub repo: `Arvind8454/VIT`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
5. **Click "Deploy"** 🚀

Your Streamlit app will be live at: `https://your-app-name.streamlit.app`

### Option 2: Deploy Flask + Streamlit to Railway (Current Setup)

Already configured! Go to https://railway.app and connect your GitHub repo.

---

## Configuration

- **Streamlit config**: `.streamlit/config.toml`
- **Requirements**: `streamlit_requirements.txt` (or use main `requirements.txt`)

## Features

Your app includes:

- ✅ ViT (Vision Transformer) image classification
- ✅ Real-time predictions
- ✅ XAI explanations (Grad-CAM, Attention, Integrated Gradients)
- ✅ Webcam live detection
- ✅ Model comparisons

## Environment Variables (if needed)

Set in Streamlit Cloud dashboard:

```
SECRET_KEY=your-secret-key
MONGO_URI=your-mongodb-uri (optional)
```

---

**Ready to deploy!** Choose your platform above.
