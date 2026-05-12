# Streamlit Cloud Deployment Guide

## 🚨 IMPORTANT: OpenCV Fix Applied

**The original app had OpenCV import errors on Streamlit Cloud.** I've created fixes:

### ✅ **Fixed Issues:**

- Changed `opencv-python` → `opencv-python-headless` (compatible with cloud platforms)
- Removed `streamlit-webrtc` (causing webcam conflicts)
- Created `streamlit_app_simple.py` (Streamlit Cloud compatible)

---

## Setup Instructions

### Option 1: Deploy Simplified Streamlit App (Recommended for Streamlit Cloud)

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in with GitHub**
3. **Click "Deploy an app"**
4. **Fill in the details:**
   - GitHub repo: `Arvind8454/VIT`
   - Branch: `main`
   - Main file path: `app/streamlit_app_simple.py` ⭐ **(Use this instead of streamlit_app.py)**
5. **Click "Deploy"** 🚀

### Option 2: Deploy Flask + Streamlit to Railway

Already configured! Go to https://railway.app and connect your GitHub repo.

---

## Configuration

- **Streamlit config**: `.streamlit/config.toml`
- **Requirements**: `streamlit_requirements.txt` (Streamlit Cloud compatible)
- **Main app**: `app/streamlit_app_simple.py` (no webcam, no OpenCV issues)

## Features (Simplified Version)

Your app includes:

- ✅ ViT (Vision Transformer) image classification
- ✅ Image upload and real-time predictions
- ✅ XAI explanations (Grad-CAM, Attention, Integrated Gradients)
- ✅ Clean, responsive UI
- ❌ Webcam functionality (removed to avoid OpenCV conflicts)

## Environment Variables (if needed)

Set in Streamlit Cloud dashboard:

```
SECRET_KEY=your-secret-key
```

---

## Files Changed

- ✅ `requirements.txt` - Updated OpenCV
- ✅ `streamlit_requirements.txt` - Updated OpenCV, removed webrtc
- ✅ `app/streamlit_app_simple.py` - New simplified app
- ✅ `STREAMLIT_DEPLOYMENT.md` - Updated guide

**Ready to deploy!** Use `app/streamlit_app_simple.py` for Streamlit Cloud.
