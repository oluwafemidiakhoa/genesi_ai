# 📥 Download Models from Google Drive (Manual Method)

**Problem:** Colab download cell giving errors?
**Solution:** Download directly from Google Drive to your computer!

---

## 🎯 Quick Method (Easiest!)

### Step 1: Open Google Drive in Your Browser

Go to: https://drive.google.com

### Step 2: Navigate to Your Models

Find this folder:
```
My Drive → breast_cancer_research
```

You'll see:
```
breast_cancer_research/
├── checkpoints/
│   ├── quick/
│   │   └── best_model.pt         ← Download this
│   └── full/
│       └── best_model.pt         ← Or this (if you ran full training)
└── results/
    └── variant_classifier_rf.pkl  ← Download this
```

### Step 3: Download Each File

**Download Genesis RNA Model:**
1. Go to `checkpoints/quick/` (or `checkpoints/full/`)
2. Right-click on `best_model.pt`
3. Click "Download"
4. File downloads to your Downloads folder

**Download Classifier:**
1. Go to `results/`
2. Right-click on `variant_classifier_rf.pkl`
3. Click "Download"
4. File downloads to your Downloads folder

---

## ✅ Verify Downloads

Check your Downloads folder. You should have:
- ✅ `best_model.pt` (~45-180 MB)
- ✅ `variant_classifier_rf.pkl` (~5-10 MB)

---

## 🌐 Upload to Hugging Face Space

### Method 1: Web Interface (Recommended)

**Step 1: Go to your Space**
https://huggingface.co/spaces/mgbam/genesis-rna-brca-classifier

**Step 2: Upload best_model.pt**
1. Click "Files" tab
2. Click "Add file" → "Upload files"
3. Click "Choose file" or drag `best_model.pt` from Downloads folder
4. In "Commit message": `Add trained Genesis RNA model`
5. **IMPORTANT:** In "Path" field at bottom, type: `models/`
6. Click "Commit changes to main"

**Wait 1-2 minutes for upload to complete**

**Step 3: Upload variant_classifier_rf.pkl**
1. Click "Add file" → "Upload files" again
2. Click "Choose file" or drag `variant_classifier_rf.pkl` from Downloads
3. In "Commit message": `Add trained classifier`
4. **IMPORTANT:** In "Path" field, type: `models/`
5. Click "Commit changes to main"

**Step 4: Wait for Space to Rebuild**
- Space will show "Building" status (~3-5 minutes)
- When it shows "Running", you're done!

**Step 5: Test Your Deployment**
1. Go back to your Space's main page
2. Enter variant: `c.5266dupC`
3. Select gene: `BRCA1`
4. Click "Predict Pathogenicity"
5. Expected result: **Pathogenic** with >95% confidence

---

## 🔧 Troubleshooting

### Problem: "Can't find models/ folder in Hugging Face"

**Solution:** The `models/` folder doesn't need to exist first. When you type `models/` in the "Path" field during upload, Hugging Face will automatically create it.

**Steps:**
1. Click "Add file" → "Upload files"
2. Select your file
3. In the "Path" field at the bottom (looks like: `space root/___`), type: `models/`
4. Commit changes

### Problem: "File too large to upload"

**Solution:** If `best_model.pt` is >180 MB, use Git LFS:

**Method A: Use Git LFS (for large files)**
```bash
# Install Git LFS (one time only)
git lfs install

# Clone your Space
git clone https://huggingface.co/spaces/mgbam/genesis-rna-brca-classifier
cd genesis-rna-brca-classifier

# Create models folder
mkdir models

# Copy files from Downloads
cp ~/Downloads/best_model.pt models/
cp ~/Downloads/variant_classifier_rf.pkl models/

# Commit and push
git add models/
git commit -m "Add trained models"
git push
```

**Method B: Compress first**
1. Right-click `best_model.pt` in Downloads folder
2. "Send to" → "Compressed (zipped) folder"
3. Upload the `.zip` file to Hugging Face
4. Add unzip step to Space (I can help with this)

### Problem: "Downloads folder is empty"

**Solution:** Check these locations:
- Windows: `C:\Users\YourName\Downloads`
- Mac: `~/Downloads` or `/Users/YourName/Downloads`

Or search your computer for:
- `best_model.pt`
- `variant_classifier_rf.pkl`

### Problem: "Files not in Google Drive"

**Solution:** Re-run the Colab cells that create the models:

**For quick training:**
- Run Cell 9 (Quick Training) - creates `checkpoints/quick/best_model.pt`
- Run Cell 24 (Classifier) - creates `results/variant_classifier_rf.pkl`

**For full training:**
- Run Cell 14 (Full Training) - creates `checkpoints/full/best_model.pt`
- Run Cell 24 (Classifier) - creates `results/variant_classifier_rf.pkl`

---

## 📊 File Size Reference

**Normal file sizes:**
- `best_model.pt` (quick/small): 45-60 MB
- `best_model.pt` (full/base): 140-180 MB
- `variant_classifier_rf.pkl`: 5-10 MB

If files are **much smaller** (<1 MB), training may not have completed successfully.

---

## 🎯 Alternative: Use Hugging Face CLI

If web upload keeps failing, use the CLI:

### Step 1: Install Hugging Face CLI
```bash
pip install huggingface_hub
```

### Step 2: Login
```bash
huggingface-cli login
```
(Enter your Hugging Face token when prompted)

### Step 3: Upload Files
```bash
# Navigate to Downloads folder
cd ~/Downloads  # Mac/Linux
cd C:\Users\YourName\Downloads  # Windows

# Upload files
huggingface-cli upload mgbam/genesis-rna-brca-classifier best_model.pt models/best_model.pt
huggingface-cli upload mgbam/genesis-rna-brca-classifier variant_classifier_rf.pkl models/variant_classifier_rf.pkl
```

---

## ✅ Success Checklist

After upload, verify:
- [ ] Space shows "Running" status (not "Building")
- [ ] Files visible at: https://huggingface.co/spaces/mgbam/genesis-rna-brca-classifier/tree/main/models
- [ ] Test prediction returns result without errors
- [ ] Confidence score is >90%

---

## 🎊 You're Done!

Once both files are uploaded and Space is running:
1. ✅ Your trained models are deployed
2. ✅ Anyone can use your Space to predict variant pathogenicity
3. ✅ You're ready to share your work!

---

## 📞 Still Having Issues?

**Common fixes:**
1. **Refresh Google Drive** - Sometimes files take a minute to appear
2. **Check Google Drive storage** - Need enough space for models
3. **Try incognito browser** - Sometimes caching causes issues
4. **Use different browser** - Chrome works best for Drive downloads

**If all else fails:**
1. Copy files from Colab to local using: `!cp /content/drive/MyDrive/breast_cancer_research/checkpoints/quick/best_model.pt /content/`
2. Right-click file in Colab file browser (left sidebar)
3. Download from Colab file browser instead of using download cell

---

**🎗️ Your models are ready. Just download and upload. You've got this!** 🚀
