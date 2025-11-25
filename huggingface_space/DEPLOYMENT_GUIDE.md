# 🚀 Deploying Genesis RNA to Hugging Face Spaces

This guide will help you deploy your Genesis RNA BRCA variant classifier to Hugging Face Spaces.

## 📋 Prerequisites

1. **Hugging Face Account**
   - Create account at: https://huggingface.co/join
   - Verify your email

2. **Git and Git LFS**
   - Install Git: https://git-scm.com/downloads
   - Install Git LFS: https://git-lfs.github.com/

3. **Hugging Face CLI**
   ```bash
   pip install huggingface_hub[cli]
   huggingface-cli login
   ```

## 🎯 Step-by-Step Deployment

### Step 1: Create a New Space

1. Go to https://huggingface.co/new-space
2. Fill in the details:
   - **Space name**: `genesis-rna-brca-classifier`
   - **License**: MIT
   - **Space SDK**: Gradio
   - **Space hardware**: CPU Basic (free) or GPU (paid)
   - **Visibility**: Public or Private

3. Click **Create Space**

### Step 2: Clone Your Space Repository

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/genesis-rna-brca-classifier
cd genesis-rna-brca-classifier
```

### Step 3: Copy Files to Space

Copy these files from `huggingface_space/` directory:

```bash
# Copy main files
cp ../huggingface_space/app.py .
cp ../huggingface_space/requirements.txt .
cp ../huggingface_space/README.md .
```

### Step 4: Add Model Files (Optional)

If you want to use the actual trained model (not mock predictions):

```bash
# Enable Git LFS for large files
git lfs install
git lfs track "*.pt"
git lfs track "*.pkl"

# Copy model files
mkdir models
cp /path/to/your/best_model.pt models/
cp /path/to/your/variant_classifier_rf.pkl models/
```

Update `app.py` to load your actual models:

```python
# At the top of app.py, add:
MODEL_PATH = "models/best_model.pt"
CLASSIFIER_PATH = "models/variant_classifier_rf.pkl"

# Load models on startup
from genesis_rna import GenesisRNAModel
from genesis_rna.tokenization import RNATokenizer

model = GenesisRNAModel.from_pretrained(MODEL_PATH, device='cpu')
tokenizer = RNATokenizer()
classifier = joblib.load(CLASSIFIER_PATH)
```

### Step 5: Commit and Push

```bash
git add .
git commit -m "Initial deployment of Genesis RNA BRCA classifier"
git push
```

### Step 6: Wait for Build

- Hugging Face will automatically build your Space
- Check the build logs at: https://huggingface.co/spaces/YOUR_USERNAME/genesis-rna-brca-classifier
- Build typically takes 2-5 minutes

### Step 7: Test Your Space

Once build completes:
- Your Space will be live at: https://huggingface.co/spaces/YOUR_USERNAME/genesis-rna-brca-classifier
- Test all features: single prediction, batch analysis, search

## 🎨 Customization Options

### Change Theme

In `app.py`, modify the theme:

```python
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    # or: Soft(), Glass(), Base()
```

### Add Logo

```python
# In app.py
with gr.Blocks() as demo:
    gr.Image("logo.png", width=200)
    gr.Markdown(f"# {TITLE}")
```

### Enable Analytics

Add to README.md frontmatter:

```yaml
---
sdk_version: 4.44.0
app_file: app.py
pinned: true  # Pin to your profile
short_description: AI-powered BRCA variant classification
tags:
  - genomics
  - cancer
  - transformer
  - biology
---
```

## 💰 Hardware Options

### Free CPU (Basic)
- **Cost**: Free
- **RAM**: 16 GB
- **vCPUs**: 2
- **Storage**: 50 GB
- **Good for**: Demo, light usage

### GPU (Paid)
- **Cost**: $0.60/hour (T4) or $3.00/hour (A10G)
- **Use case**: Faster inference, many users
- **Enable in**: Space settings → Hardware

## 📊 Using Real Models

To deploy with your actual Genesis RNA model:

### Option A: Small Model (Recommended)

1. **Train small model** (10M params, ~40MB)
   ```bash
   python -m genesis_rna.train_pretrain \
       --model_size small \
       --num_epochs 10 \
       --output_dir models/small
   ```

2. **Copy to Space**
   ```bash
   cp models/small/best_model.pt huggingface_space/models/
   cp variant_classifier_rf.pkl huggingface_space/models/
   ```

3. **Update app.py** to load models (see Step 4 above)

### Option B: Use Hugging Face Model Hub

1. **Upload model to Hub**
   ```python
   from huggingface_hub import HfApi

   api = HfApi()
   api.upload_folder(
       folder_path="models/",
       repo_id="YOUR_USERNAME/genesis-rna-brca",
       repo_type="model"
   )
   ```

2. **Load in app.py**
   ```python
   from huggingface_hub import hf_hub_download

   model_path = hf_hub_download(
       repo_id="YOUR_USERNAME/genesis-rna-brca",
       filename="best_model.pt"
   )
   ```

## 🔒 Handling Secrets

For API keys or sensitive data:

1. **Go to Space settings** → Secrets
2. **Add secret**: Name = `API_KEY`, Value = `your-key`
3. **Access in app.py**:
   ```python
   import os
   api_key = os.environ.get("API_KEY")
   ```

## 📈 Monitoring Usage

Track your Space usage:

1. Go to Space → Analytics
2. View:
   - Daily active users
   - Total runs
   - Average response time
   - Error rate

## 🐛 Troubleshooting

### Build Fails

**Problem**: Build errors

**Solutions**:
- Check requirements.txt (compatible versions?)
- View build logs for specific error
- Test locally first: `python app.py`

### Out of Memory

**Problem**: Space crashes with OOM

**Solutions**:
- Reduce model size (use small instead of base)
- Batch predictions in smaller chunks
- Upgrade to GPU hardware
- Use model quantization

### Slow Inference

**Problem**: Predictions take too long

**Solutions**:
- Cache model in memory (load once)
- Use smaller model
- Upgrade to GPU
- Optimize batch processing

### Model Not Found

**Problem**: `FileNotFoundError: models/best_model.pt`

**Solutions**:
- Ensure Git LFS tracked .pt files
- Push large files correctly: `git lfs push --all origin main`
- Check file size limit (5GB per file on Spaces)

## 🚀 Advanced: Production Deployment

For serious production use:

### 1. Add Rate Limiting

```python
from gradio.components import State
import time

user_requests = {}

def rate_limit(user_id):
    now = time.time()
    if user_id in user_requests:
        if now - user_requests[user_id] < 1:  # 1 req/sec
            return False
    user_requests[user_id] = now
    return True
```

### 2. Add Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_variant(variant_id, ...):
    logger.info(f"Prediction request: {variant_id}")
    # ... prediction logic
    logger.info(f"Prediction complete: {prediction}")
```

### 3. Error Handling

```python
def predict_variant(variant_id, ...):
    try:
        # ... prediction logic
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return f"Error: {str(e)}"
```

### 4. Add Authentication (Optional)

For private deployment:

```python
demo.launch(auth=("username", "password"))
```

Or use Hugging Face OAuth:

```python
demo.launch(auth="huggingface")
```

## 📚 Resources

- **Gradio Docs**: https://gradio.app/docs/
- **HF Spaces Docs**: https://huggingface.co/docs/hub/spaces
- **Git LFS Guide**: https://git-lfs.github.com/
- **Example Spaces**: https://huggingface.co/spaces

## 🎉 You're Done!

Your Genesis RNA BRCA classifier is now live on Hugging Face Spaces!

**Share your Space:**
- Direct link: https://huggingface.co/spaces/YOUR_USERNAME/genesis-rna-brca-classifier
- Embed on website: Get embed code from Space settings
- Social media: Share with #GenesisRNA #BreastCancerResearch

---

**Need help?** Open an issue on GitHub or ask in Hugging Face forums!
