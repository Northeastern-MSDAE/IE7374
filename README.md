

# QA Attention Analysis Pipeline

This repository implements a complete pipeline to analyze and compare transformer attention patterns across two domains: Medical and Legal QA.

## Project Structure

- `notebooks/`
  - `attention_analysis_domain_comparison.ipynb`: 
    - Data loading, sampling, cleaning, and tokenization
    - Zero-shot tagging and keyword extraction
    - Computation of attention diagnostics
    - PCA/t-SNE and distribution plots
- `notebooks/utils`
  - util functions
- `requirements.txt`: Python dependencies
- `README.md`: Project overview & setup instructions

## Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Northeastern-MSDAE/IE7500.git
   cd IE7500
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Pipeline Steps

1. **Data Preparation**
   - Load MedMCQA and Legal QA datasets
   - Sample a balanced subset (configurable `N`)
   - Clean text (remove HTML/markdown, Q/A prefixes)
   - Tokenize and map subword tokens back to words
2. **Tagging & Keyword Extraction**
   - Zero-shot classification with HuggingFace pipeline
   - Compute `auto_tags` for each question
   - Extract domain-relevant keywords via TF-IDF or YAKE
   - Compute salience metrics: `TagSalience` and `KeywordSalience`
3. **Attention Metrics**
   - Run T5 model to obtain decoder self-attention and cross-attention tensors
   - Compute per-question diagnostics:
     - **Entropy** of attention heads
     - **Volatility** (max-min) across heads
     - **Span** (fraction of non-zero attention)
     - **Head Agreement** (Jensen–Shannon divergence)
     - **SAAS** (Self vs. Auto Attention Score)
4. **Statistical Analysis**
   - Mann–Whitney U tests to compare distributions across domains
   - Summarize means, std deviations, and p-values
5. **Visualization**
   - Histograms of each metric by domain
   - PCA/t-SNE plots of flattened cross-attention vectors
   - Heatmaps for example questions

## Results

- Summary statistics and significance tests are in `reports/`
- Plots are saved in `figures/`

## Configuration

- Modify sampling size `N` in `data_prep.ipynb`
- Choose device (CPU/GPU) via `--device` flag in scripts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/...`)
3. Commit your changes (`git commit -am 'Add ...'`)
4. Push to the branch (`git push origin feature/...`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See `LICENSE` for details.
