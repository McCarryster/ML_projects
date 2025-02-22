Metrics I used for evaluation:
    1. Accuracy
    2. BLEU
    3. Rogue: rouge = PyRouge(rouge_n=(1, 2), rouge_l=True)
        • rouge_n=(1, 2) implies that the ROUGE score will take into account both individual words (unigrams) and pairs of words (bigrams) when comparing the generated and reference texts
        • rouge_l=True measures the longest common subsequence (LCS) between the generated and reference texts. The LCS is the longest sequence of words that appear in both texts in the same order, but not necessarily contiguously
    4. METEOR: ...