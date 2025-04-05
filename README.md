# Simplified MUSE Implementation

This is a simplified implementation of the unsupervised adversarial method for bilingual word embeddings alignment, inspired by the [MUSE](https://github.com/facebookresearch/MUSE) project by Facebook Research.

## Overview

This implementation provides a way to align word embeddings across languages without using parallel data (dictionaries). It uses an adversarial approach with:

1. A **mapping function** that transforms embeddings from one language to another
2. A **discriminator** that tries to distinguish between real embeddings and mapped embeddings
3. **CSLS (Cross-domain Similarity Local Scaling)** for better translation retrieval

## Key Components

### 1. Unsupervised Alignment

The core of the implementation is the `unsupervised_alignment` function, which:

- Takes English and Hindi embeddings as input
- Trains a mapping function to transform English embeddings into the Hindi embedding space
- Uses adversarial training to ensure the mapped embeddings are indistinguishable from real Hindi embeddings
- Returns the aligned English embeddings

### 2. Translation Retrieval

Translation is performed using the CSLS method, which:

- Calculates similarity between an English word and all Hindi words
- Applies cross-domain similarity local scaling to improve translation quality
- Returns the top k Hindi translations for a given English word

### 3. Interactive Testing

The implementation includes an interactive testing function that allows you to:

- Enter English words and see their Hindi translations
- Test the quality of the alignment
- Exit when done

## Usage

### Small Example

The code includes a small example with a few words in English and Hindi to demonstrate the functionality:

```python
# Create small example embeddings
en_embeddings = {
    "hello": np.array([0.1, 0.2, 0.3], dtype=np.float32),
    "world": np.array([0.4, 0.5, 0.6], dtype=np.float32),
    "good": np.array([0.7, 0.8, 0.9], dtype=np.float32),
}

hi_embeddings = {
    "नमस्ते": np.array([0.15, 0.25, 0.35], dtype=np.float32),
    "दुनिया": np.array([0.45, 0.55, 0.65], dtype=np.float32),
    "अच्छा": np.array([0.75, 0.85, 0.95], dtype=np.float32),
}
```

### Full Embeddings

To use the full embeddings, you need to download the FastText embeddings:

- English: `wiki.en.vec`
- Hindi: `wiki.hi.vec`

Place these files in the same directory as the script and run it.

## Differences from Original MUSE

This implementation is simplified compared to the original MUSE project:

1. **Simpler Architecture**: Uses a basic linear mapping and discriminator
2. **No Refinement**: Omits the refinement step that uses CSLS to improve the mapping
3. **No Validation**: Doesn't include validation on a small dictionary
4. **No Orthogonal Constraint**: Doesn't enforce orthogonality on the mapping

## Requirements

- Python 3.6+
- NumPy
- PyTorch
- SciPy

## References

- [MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE)
- [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)
