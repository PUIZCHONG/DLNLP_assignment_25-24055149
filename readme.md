# NBME - Score Clinical Patient Notes

This repository contains code for the [NBME - Score Clinical Patient Notes](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes) Kaggle competition. The competition focuses on identifying clinical concepts in patient notes using natural language processing techniques.

## Project Overview

The challenge is to extract and identify specific medical concepts mentioned in clinical patient notes. The project uses transformer-based models (specifically DeBERTa-v3-large) to identify these medical concepts based on feature descriptions.

In this competition, we're tasked with detecting clinical concepts in patient notes. Healthcare professionals document patient encounters in these notes, and we need to identify which parts of the text refer to specific medical concepts (like symptoms, diagnoses, or medications). This helps automate the scoring process for medical licensing exams.

## Repository Structure

This repository contains five main Python files:

### 1. `eda.py` - Exploratory Data Analysis

This script performs comprehensive exploratory data analysis on the competition dataset:
- Analyzes word count distribution in patient history
- Examines annotation distribution in training data
- Analyzes feature word count distribution
- Visualizes the relationship between feature term counts and frequencies
- Creates visualizations for patient history word counts and annotation distributions

### 2. `preprocess.py` - Data Preprocessing

This script handles data preprocessing operations:
- Loads and merges all necessary data files
- Identifies feature types (gender, age, etc.)
- Standardizes medical terminology and abbreviations
- Processes annotation locations and handles offset corrections
- Creates cross-validation folds using GroupKFold
- Prepares training and test data for model training

### 3. `test.py` - Model Inference

This script handles model inference:
- Loads trained DeBERTa-v3-large model weights
- Performs inference on test data
- Converts token-level predictions to character-level spans
- Tunes threshold for optimal F1 score
- Generates submission file

### 4. `evaluate_oof.py` - Out-of-Fold Evaluation

This script evaluates model performance:
- Computes span-level micro-F1 score from out-of-fold predictions
- Tunes threshold for optimal performance
- Provides detailed evaluation metrics for model validation
- Can be run from the command line with various parameters

### 5. `training_kaggle.ipynb` - Model Training Notebook

This Jupyter notebook is specifically optimized for training in the Kaggle environment:
- Includes automatic path detection for Kaggle directories
- Implements complete training pipeline with cross-validation
- Features adversarial training (FGM) for improved model robustness
- Uses Focal Loss with label smoothing to handle class imbalance
- Implements automatic threshold tuning for optimal F1 score
- Includes visualizations for threshold tuning and error analysis
- Provides memory optimizations for Kaggle's GPU environment
- Contains full mixed-precision training implementation for faster training

## Solution Approach

Our solution uses the following approach to identify medical concepts in clinical notes:

1. **Data Preprocessing**: 
   - Standardize medical abbreviations (e.g., "htn" → "hypertension")
   - Handle special case annotations and adjust offsets
   - Clean up spaces in text for better boundary detection
   - Group cross-validation based on patient note numbers

2. **Model Architecture**:
   - DeBERTa-v3-large as the backbone model
   - Token classification head for span detection
   - Focal Loss with label smoothing to handle class imbalance
   - Mixed-precision training (FP16) for efficiency

3. **Advanced Training Techniques**:
   - Adversarial training using Fast Gradient Method (FGM)
   - Cosine scheduler with warmup
   - Gradient accumulation for larger effective batch sizes
   - Character-level predictions with optimal threshold tuning

4. **Post-processing**:
   - Convert token predictions to character-level spans
   - Space processing to handle word boundaries correctly
   - Threshold tuning to maximize F1 score
   - Special handling for medical notation (e.g., "yof" for "year-old female")

This approach achieved a high micro-F1 score on the competition's validation set, demonstrating effective identification of medical concepts in clinical notes.

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Pandas, NumPy, Matplotlib, Seaborn
- tqdm

### Installation

```bash
git clone https://github.com/yourusername/nbme-score-clinical-patient-notes.git
cd nbme-score-clinical-patient-notes
pip install -r requirements.txt
```

### Training on Kaggle

To train the model on Kaggle:

1. Navigate to the [NBME - Score Clinical Patient Notes](https://www.kaggle.com/code/nocharon/nbme-nlp) competition
2. Create a new notebook
3. Upload the `training_kaggle.ipynb` file or copy its contents
4. Run the notebook to train the model

The notebook will automatically detect Kaggle paths and optimize settings for the Kaggle environment.

### Running Locally

For local development and training:

1. Download the competition data from [Kaggle](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes/data)
2. Update the file paths in the scripts to match your local setup
3. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```

### Running EDA

```bash
python eda.py
```

### Model Training (Local)

```bash
python train.py  # You might need to adapt the training_kaggle.ipynb to a Python script
```

### Model Inference

```bash
python test.py
```

### Evaluation

```bash
python evaluate_oof.py --oof_pkl path/to/oof_df_0.pkl --train_csv path/to/train.csv
```

## Performance

The model achieves a competitive micro-F1 score by:
- Properly handling medical terminology and abbreviations
- Correctly identifying spans of text that correspond to medical concepts
- Using threshold tuning to optimize the F1 score
- Employing adversarial training for better generalization
- Applying special handling for medical notation patterns

### Key Results

- Cross-validation F1 score: ~0.89 (with optimal threshold)
- Performance varies by annotation length, with mid-length annotations having the highest F1 scores
- Common error types include:
  - Partial matches (identifying only part of the concept)
  - Boundary errors (including too much or too little text)
  - False negatives on rare or complex medical terms

## Competition Context

This competition was hosted by the National Board of Medical Examiners (NBME), which develops and manages assessments of healthcare professionals. The goal was to automatically detect the location of clinical concepts in medical licensing exam responses, helping to streamline the exam scoring process.

The ability to accurately identify medical concepts in text has broader applications in:
- Electronic health record (EHR) analysis
- Medical literature review
- Clinical decision support systems
- Medical education and assessment

## Model Architecture

The project uses a DeBERTa-v3-large transformer model with a token classification head to identify spans of text corresponding to medical concepts. The model processes patient notes along with feature descriptions to produce token-level predictions, which are then converted to character-level spans.

## Competition Metrics

The competition is evaluated using span-level micro-F1 score. This measures how well the model identifies the exact character spans that correspond to each medical concept.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [NBME](https://www.nbme.org/) for providing the dataset
- Kaggle for hosting the competition