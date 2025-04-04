import numpy as np
import io

def load_embeddings(file_path):
    """Loads FastText embeddings from a file."""
    embeddings = {}
    with io.open(file_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)  # Skip header
        for line in f:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def limit_vocabulary(embeddings, top_n=100000):
    """Limits the vocabulary to the top N most frequent words."""
    sorted_words = sorted(embeddings.items(), key=lambda x: len(x[1]), reverse=True)
    limited_embeddings = {}
    for i, (word, vector) in enumerate(sorted_words):
        if i < top_n:
            limited_embeddings[word] = vector
        else:
            break
    return limited_embeddings

def load_bilingual_lexicon(file_path, en_embeddings, hi_embeddings):
    """Loads a bilingual lexicon from a file and filters it based on available embeddings."""
    en_words = []
    hi_words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 2:
                continue
            en_word, hi_word = line.strip().split()
            if en_word in en_embeddings and hi_word in hi_embeddings:
                en_words.append(en_word)
                hi_words.append(hi_word)
    return en_words, hi_words

def procrustes_alignment(en_embeddings, hi_embeddings, en_lexicon, hi_lexicon):
    """Performs Procrustes alignment."""
    en_matrix = np.array([en_embeddings[word] for word in en_lexicon])
    hi_matrix = np.array([hi_embeddings[word] for word in hi_lexicon])

    # en_mean = np.mean(en_matrix, axis=0, keepdims=True)
    # hi_mean = np.mean(hi_matrix, axis=0, keepdims=True)
    # en_matrix = en_matrix - en_mean
    # hi_matrix = hi_matrix - hi_mean

    en_matrix = en_matrix / np.linalg.norm(en_matrix, axis=1)[:, np.newaxis]
    hi_matrix = hi_matrix / np.linalg.norm(hi_matrix, axis=1)[:, np.newaxis]

    U, _, Vt = np.linalg.svd(np.dot(en_matrix.T, hi_matrix))
    W = np.dot(U, Vt)

    return W

def apply_mapping(en_embeddings, W):
    """Applies the learned mapping to English embeddings."""
    aligned_en_embeddings = {}
    for word, vector in en_embeddings.items():
        aligned_en_embeddings[word] = np.dot(vector, W)
    return aligned_en_embeddings

def translate_word(en_word, aligned_en_embeddings, hi_embeddings, k=5):
    """Translates an English word to Hindi."""
    if en_word not in aligned_en_embeddings:
        return []

    en_vector = aligned_en_embeddings[en_word]
    similarities = {}
    for hi_word, hi_vector in hi_embeddings.items():
        similarities[hi_word] = np.dot(en_vector, hi_vector) / (np.linalg.norm(en_vector) * np.linalg.norm(hi_vector))

    sorted_translations = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_translations[:k]]

def evaluate_translation(aligned_en_embeddings, hi_embeddings, test_lexicon_path, k=5):
    """Evaluates translation accuracy using Precision@k."""
    test_en_words, test_hi_words = load_bilingual_lexicon(test_lexicon_path, en_embeddings, hi_embeddings)

    precision_at_1 = 0
    precision_at_5 = 0

    for en_word, hi_word in zip(test_en_words, test_hi_words):
        translations = translate_word(en_word, aligned_en_embeddings, hi_embeddings, k)
        if hi_word in translations[:1]:
            precision_at_1 += 1
        if hi_word in translations:
            precision_at_5 += 1

    precision_at_1 /= len(test_en_words)
    precision_at_5 /= len(test_en_words)

    return precision_at_1, precision_at_5

def compute_cosine_similarity(word1, word2, embeddings1, embeddings2, W=None):
    """Computes cosine similarity between two words."""
    if W is not None:
      embeddings1[word1] = np.dot(embeddings1[word1], W)

    vec1 = embeddings1[word1]
    vec2 = embeddings2[word2]
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def ablation_study(en_embeddings, hi_embeddings, lexicon_path, test_lexicon_path, sizes=[5000, 10000, 20000]):
    """Conducts an ablation study for different lexicon sizes."""
    results = {}
    for size in sizes:
        en_lexicon, hi_lexicon = load_bilingual_lexicon(lexicon_path, en_embeddings, hi_embeddings)
        en_lexicon = en_lexicon[:size]
        hi_lexicon = hi_lexicon[:size]

        W = procrustes_alignment(en_embeddings, hi_embeddings, en_lexicon, hi_lexicon)
        aligned_en_embeddings = apply_mapping(en_embeddings, W)

        precision_1, precision_5 = evaluate_translation(aligned_en_embeddings, hi_embeddings, test_lexicon_path, 5)
        results[size] = (precision_1, precision_5)

    return results

# Main execution
en_embeddings = load_embeddings("wiki.en.vec") #Replace with your path
hi_embeddings = load_embeddings("wiki.hi.vec") #Replace with your path

en_embeddings = limit_vocabulary(en_embeddings)
hi_embeddings = limit_vocabulary(hi_embeddings)

en_lexicon, hi_lexicon = load_bilingual_lexicon("en-hi.txt", en_embeddings, hi_embeddings)

W = procrustes_alignment(en_embeddings, hi_embeddings, en_lexicon, hi_lexicon)
aligned_en_embeddings = apply_mapping(en_embeddings, W)

precision_1, precision_5 = evaluate_translation(aligned_en_embeddings, hi_embeddings, "en-hi.txt", 5)
print(f"Precision@1: {precision_1}")
print(f"Precision@5: {precision_5}")

sim = compute_cosine_similarity("hello","नमस्ते", en_embeddings, hi_embeddings, W)
print(f"Cosine similarity: {sim}")

ablation_results = ablation_study(en_embeddings, hi_embeddings, "en-hi.txt", "en-hi.txt")
print("Ablation Study Results:")
for size, (p1, p5) in ablation_results.items():
    print(f"Size: {size}, Precision@1: {p1}, Precision@5: {p5}")
