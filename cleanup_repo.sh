#!/bin/bash
# Git Repository Cleanup Script
# Removes duplicate, old, and unnecessary documentation files

echo "==============================================="
echo "Genesis RNA Repository Cleanup"
echo "==============================================="
echo ""
echo "This will remove old/duplicate documentation files from git tracking."
echo "Files will still exist on disk, just not tracked by git."
echo ""

# Files to remove from git (but keep on disk)
FILES_TO_REMOVE=(
    # Old/duplicate documentation
    "README_OLD_AI_VERSION.md"
    "README_HUMAN.md"
    "ALL_FIXES_SUMMARY.md"
    "FINAL_FIX_SUMMARY.md"
    "NOTEBOOK_FIX_COMPLETE.md"
    "COMPLETE_PROJECT_SUMMARY.md"
    "COMPLETE_REAL_DATA_GUIDE.md"

    # Redundant quick starts (keep only one)
    "QUICK_START_CANCER_CURE.md"
    "QUICKSTART_REAL_DATA.md"
    "READY_TO_LAUNCH.md"
    "READY_TO_RUN.md"

    # Temporary fix documentation (issues now fixed)
    "FIX_COLAB_NOTEBOOK.md"
    "FIX_DOMAIN_SHIFT.md"
    "COLAB_DESIGNER_FIX.md"
    "CHECKPOINT_FIX_NOTES.md"

    # Marketing/launch content (not needed in repo)
    "LINKEDIN_POST.md"
    "TWITTER_THREADS.md"
    "MEDIUM_ARTICLE.md"
    "PRESS_RELEASE.md"
    "MY_CONTRIBUTION_TO_CURING_CANCER.md"
    "OLUWAFEMI_BIO.md"
    "SHARE_YOUR_WORK.md"

    # Temporary instruction files
    "ADD_TO_COLAB_DOWNLOAD_CELL.txt"
    "COLAB_RELOAD_CELL.txt"
    "COPY_PASTE_THIS_CODE.txt"
    "QUICK_DATA_DOWNLOAD.txt"
    "QUICK_UPLOAD_GUIDE.md"
    "DOWNLOAD_FROM_GOOGLE_DRIVE.md"

    # Redundant guides (covered by main docs)
    "DATA_COLLECTION_GUIDE.md"
    "HUGGINGFACE_SPACE_GUIDE.md"
    "COLAB_TO_HUGGINGFACE_DEPLOYMENT.md"
    "NEXT_STEPS_DEPLOYMENT.md"
    "FINAL_LAUNCH_CHECKLIST.md"
    "CAN_I_RUN_COLAB.md"

    # Old response files (kept only REDDIT_RESPONSE_FINAL.md)
    "REDDIT_RESPONSE.md"
    "REDDIT_RESPONSE_OVERFITTING.md"
    "ADDRESSING_DATA_LEAKAGE_CONCERN.md"

    # Upgrade notes (covered in IMPROVEMENTS.md)
    "UPGRADE_SUMMARY.md"
    "REAL_EMBEDDINGS_UPGRADE.md"
    "CANCER_RESEARCH_ENHANCEMENTS.md"
    "NOTEBOOK_TROUBLESHOOTING.md"
)

echo "Files to remove from git tracking:"
echo ""
for file in "${FILES_TO_REMOVE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
read -p "Continue with cleanup? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Removing files from git (files stay on disk)..."

    for file in "${FILES_TO_REMOVE[@]}"; do
        if [ -f "$file" ]; then
            git rm --cached "$file" 2>/dev/null
            echo "  ✓ Removed: $file"
        fi
    done

    echo ""
    echo "Adding to .gitignore..."

    # Add to .gitignore
    cat >> .gitignore << 'EOF'

# Old/duplicate documentation (cleaned up 2025-01-27)
README_OLD_AI_VERSION.md
README_HUMAN.md
ALL_FIXES_SUMMARY.md
FINAL_FIX_SUMMARY.md
NOTEBOOK_FIX_COMPLETE.md
COMPLETE_PROJECT_SUMMARY.md
COMPLETE_REAL_DATA_GUIDE.md
QUICK_START_CANCER_CURE.md
QUICKSTART_REAL_DATA.md
READY_TO_LAUNCH.md
READY_TO_RUN.md
FIX_*.md
COLAB_DESIGNER_FIX.md
CHECKPOINT_FIX_NOTES.md
LINKEDIN_POST.md
TWITTER_THREADS.md
MEDIUM_ARTICLE.md
PRESS_RELEASE.md
MY_CONTRIBUTION_TO_CURING_CANCER.md
OLUWAFEMI_BIO.md
SHARE_YOUR_WORK.md
ADD_TO_COLAB_DOWNLOAD_CELL.txt
COLAB_RELOAD_CELL.txt
COPY_PASTE_THIS_CODE.txt
QUICK_DATA_DOWNLOAD.txt
QUICK_UPLOAD_GUIDE.md
DOWNLOAD_FROM_GOOGLE_DRIVE.md
DATA_COLLECTION_GUIDE.md
HUGGINGFACE_SPACE_GUIDE.md
COLAB_TO_HUGGINGFACE_DEPLOYMENT.md
NEXT_STEPS_DEPLOYMENT.md
FINAL_LAUNCH_CHECKLIST.md
CAN_I_RUN_COLAB.md
REDDIT_RESPONSE.md
REDDIT_RESPONSE_OVERFITTING.md
ADDRESSING_DATA_LEAKAGE_CONCERN.md
UPGRADE_SUMMARY.md
REAL_EMBEDDINGS_UPGRADE.md
CANCER_RESEARCH_ENHANCEMENTS.md
NOTEBOOK_TROUBLESHOOTING.md
EOF

    echo ""
    echo "✓ Cleanup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review changes: git status"
    echo "2. Commit: git commit -m 'Clean up old/duplicate documentation files'"
    echo "3. Push: git push origin main"
    echo ""
    echo "Files removed from git: ${#FILES_TO_REMOVE[@]}"
    echo "Repository will be much cleaner!"
else
    echo ""
    echo "Cleanup cancelled."
fi
