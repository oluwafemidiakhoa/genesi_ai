# Genesis RNA Project Visualizations

This directory contains publication-quality visualizations showcasing the Genesis RNA project achievements.

## Generated Visualizations

### 1. genesis_rna_summary.png
**Main Project Infographic**
- Key achievements (100% accuracy, 50K+ sequences, 256-dim embeddings)
- Performance comparison (Baseline vs Genesis RNA)
- Data sources (Ensembl, ClinVar)
- Model architecture overview
- Training efficiency with AST
- Clinical impact areas
- Confusion matrix
- Technology stack

**Use for:** Social media posts, presentations, project overview

---

### 2. performance_timeline.png
**Evolution Over Time**
- Accuracy progression (67% → 100%)
- Feature richness growth (2 → 256 features)
- Training efficiency impact
- Clinical readiness score

**Use for:** Technical presentations, blog posts, showing improvement

---

### 3. data_statistics.png
**Data Analysis Dashboard**
- Training data distribution (ncRNA types)
- ClinVar variant distribution
- Sequence length distribution
- Gene distribution (BRCA1/BRCA2)
- Embedding visualization (t-SNE)
- Data quality metrics

**Use for:** Data transparency, reproducibility documentation

---

### 4. clinical_impact.png
**Clinical Applications**
- VUS reclassification potential (40% → 10%)
- Clinical workflow integration
- Patient benefit metrics
- Clinical validation (radar chart)
- Research applications

**Use for:** Clinical presentations, grant applications, impact statements

---

## How to Use

### For Social Media
- **LinkedIn:** Use `genesis_rna_summary.png` in post about project launch
- **Twitter:** Use `performance_timeline.png` to show evolution
- **Medium Article:** Include all 4 visualizations throughout article

### For Presentations
1. **Introduction slide:** genesis_rna_summary.png
2. **Methods slide:** data_statistics.png
3. **Results slide:** performance_timeline.png
4. **Impact slide:** clinical_impact.png

### For Publications
- **Figure 1:** Model architecture and data (from summary)
- **Figure 2:** Performance comparison (timeline)
- **Figure 3:** Clinical impact and applications

### For GitHub README
Embed as images:
```markdown
![Genesis RNA Summary](visualizations/genesis_rna_summary.png)
```

---

## Regenerating Visualizations

To create updated visualizations:

```bash
# All visualizations
python scripts/create_project_visualization.py --type all

# Individual visualizations
python scripts/create_project_visualization.py --type summary
python scripts/create_project_visualization.py --type timeline
python scripts/create_project_visualization.py --type data
python scripts/create_project_visualization.py --type clinical

# Custom output directory
python scripts/create_project_visualization.py --type all --output_dir custom_viz
```

---

## Customization

Edit `scripts/create_project_visualization.py` to:
- Update metrics and numbers
- Change color schemes (COLORS dictionary)
- Modify layout and sections
- Add new visualizations

---

## File Details

| File | Size | Resolution | Format | DPI |
|------|------|------------|--------|-----|
| genesis_rna_summary.png | ~2-3 MB | 4800x3600 | PNG | 300 |
| performance_timeline.png | ~1-2 MB | 4200x3000 | PNG | 300 |
| data_statistics.png | ~1-2 MB | 4200x3000 | PNG | 300 |
| clinical_impact.png | ~1-2 MB | 4200x3000 | PNG | 300 |

All images are high-resolution (300 DPI) suitable for:
- Print publications
- Conference posters
- High-quality presentations
- Social media (will be resized automatically)

---

## Color Palette

The visualizations use this color scheme:

- **Primary (Blue):** #2E86AB - Main elements, model components
- **Secondary (Purple):** #A23B72 - Data sources, secondary info
- **Success (Green):** #06A77D - Positive metrics, benign variants
- **Warning (Orange):** #F18F01 - VUS, efficiency metrics
- **Danger (Red):** #C73E1D - Pathogenic variants
- **Pathogenic:** #D32F2F - Clinical pathogenic
- **Benign:** #388E3C - Clinical benign
- **Neutral (Gray):** #757575 - Baselines, comparisons

---

## Attribution

Created for the Genesis RNA project by Oluwafemi Idiakhoa.

**Citation:**
```bibtex
@software{genesis_rna_2025,
  title={Genesis RNA: A Foundation Model for Cancer Variant Classification},
  author={Oluwafemi Idiakhoa},
  year={2025},
  url={https://github.com/oluwafemidiakhoa/genesi_ai}
}
```

---

## License

MIT License - Free to use with attribution

---

**Built for breast cancer research**
