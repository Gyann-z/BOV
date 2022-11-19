# Beyond OCR + VQA: Involving OCR into the Flow for Robust and Accurate TextVQA

This is the implementation of the 2021 ACM MM Oral paper "[Beyond OCR + VQA: Involving OCR into the Flow for Robust and Accurate TextVQA](https://dl.acm.org/doi/10.1145/3474085.3475606)". The code is based on [M4C](https://github.com/facebookresearch/mmf). Please refer to it for the training and evaluation details.

## Data
### Visually Enhanced Text Embedding
The implementation of TVS is modified based on [ASTER](https://github.com/ayumiymk/aster.pytorch). The TVS features (Visually derived text embeddings) are released as follows.
|Datasets|OCR TVS Features|
|--------|--------|
|TextVQA|[TextVQA TVS Rosetta Features](https://drive.google.com/file/d/1B3eQH8CSPLDiUR_FOJQ3djppwHr1Qlu3/view?usp=share_link)|
|ST-VQA|[ST-VQA TVS Rosetta Features](https://drive.google.com/file/d/11YWBN5hueciPulCCK5LsAI6FSkf7lLzv/view?usp=share_link)|

### Semantically Oriented Object Embedding
The implementation of SEO-FRCN is modified based on [BUTD](https://github.com/MILVLG/bottom-up-attention.pytorch).
