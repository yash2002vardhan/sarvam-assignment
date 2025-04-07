# Bilingual Word Embedding Alignment

This repository contains implementations of both supervised and unsupervised approaches for aligning word embeddings between English and Hindi languages. The goal is to enable cross-lingual word translation by mapping words from one language's embedding space to another.

## Files Required

To run this project, you'll need the following files:

- `wiki.en.vec`: English word embeddings file (FastText format)
  - Download: [wiki.en.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec)
- `wiki.hi.vec`: Hindi word embeddings file (FastText format)
  - Download: [wiki.hi.vec](https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.vec)
- `en-hi.txt`: Training bilingual lexicon for supervised approach
  - Download: [en-hi.txt](https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt)
- `en-hi-test.txt`: Test bilingual lexicon for evaluation
  - Download: [en-hi.5000-6500.txt](https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.5000-6500.txt)

Please place these files in the project directory before running the notebooks.

## Supervised Approach (`supervised.ipynb`)

The supervised approach uses a bilingual lexicon (dictionary) to learn the mapping between English and Hindi word embeddings. Here's what the notebook does:

1. **Loading Embeddings**: Loads pre-trained FastText word embeddings for both languages
2. **Bilingual Lexicon Processing**: Processes the training dictionary to create aligned word pairs
3. **Procrustes Alignment**: Implements the Procrustes analysis to find the optimal linear transformation between embedding spaces
4. **Translation**: Provides functions to translate words using cosine similarity in the aligned space
5. **Evaluation**: Implements Precision@k metrics to evaluate translation quality
6. **Ablation Study**: Analyzes performance with different sizes of training lexicons

The supervised approach achieves:
- Precision@1: ~59.34%
- Precision@5: ~78.06%

## Optional Credit: Unsupervised Approach (`unsupervised.ipynb`)

The unsupervised approach uses an adversarial training strategy to align word embeddings without using a bilingual dictionary. Here's what the notebook implements:

1. **Generator-Discriminator Architecture**:
   - Generator: Learns a linear mapping between source and target embedding spaces
   - Discriminator: Tries to distinguish between real target embeddings and fake ones

2. **Key Components**:
   - Orthogonal initialization and regularization for the mapping matrix
   - Cross-domain Similarity Local Scaling (CSLS) to reduce hubness problem
   - Adversarial training with early stopping

3. **Training Process**:
   - Alternates between training the discriminator and generator
   - Uses BCE loss for adversarial training
   - Implements early stopping based on generator loss
   - Saves the best model based on performance

4. **Translation**:
   - Uses CSLS similarity for finding nearest neighbors
   - Handles out-of-vocabulary words gracefully
   - Returns top-k translation candidates

The unsupervised approach provides an alternative method that doesn't require parallel data, though it may achieve lower accuracy compared to the supervised approach.

## Usage

1. Place the required files in the project directory
2. Run either notebook based on your needs:
   - Use `supervised.ipynb` if you have a bilingual dictionary
   - Use `unsupervised.ipynb` if you don't have parallel data

## Dependencies

- Python 3.x
- NumPy
- PyTorch
- FastText embeddings

## Notes

- The performance of both approaches depends heavily on the quality and size of the word embeddings
- The supervised approach generally performs better but requires parallel data
- The unsupervised approach is more flexible but may need more tuning to achieve good results
