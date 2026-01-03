"""
Phrase Clustering module for the Crypto Narrative Pulse Tracker.

Extracts and clusters trending phrases (bigrams and trigrams) from messages
to detect emerging narratives in crypto discussions.

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

# =============================================================================
# STOPWORDS
# =============================================================================

STOPWORDS = {
    # Common English stopwords
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "this",
    "that",
    "these",
    "those",
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "about",
    "get",
    "got",
    "getting",
    "like",
    "really",
    "going",
    "gonna",
    "want",
    "know",
    "think",
    "see",
    "look",
    "make",
    "way",
    "even",
    "new",
    "now",
    "also",
    "well",
    "back",
    "much",
    "any",
    "good",
    "first",
    "last",
    "long",
    "great",
    "little",
    "own",
    "other",
    "old",
    "right",
    "big",
    "high",
    "different",
    "small",
    "large",
    "next",
    "early",
    "young",
    "important",
    "few",
    "public",
    "bad",
    "same",
    "able",
    "im",
    "dont",
    "cant",
    "wont",
    "didnt",
    "doesnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
}

# Crypto-specific terms to preserve (not filter out)
CRYPTO_PRESERVE = {
    "moon",
    "pump",
    "dump",
    "rug",
    "bullish",
    "bearish",
    "hodl",
    "fomo",
    "fud",
    "degen",
    "lfg",
    "wagmi",
    "ngmi",
    "gm",
    "alpha",
    "gem",
    "based",
    "ape",
    "diamond",
    "paper",
    "hands",
    "lambo",
    "rekt",
    "whale",
    "dip",
    "buy",
    "sell",
    "hold",
    "long",
    "short",
    "bull",
    "bear",
    "green",
    "red",
    "candle",
    "chart",
    "volume",
    "market",
    "cap",
    "price",
    "token",
    "coin",
    "crypto",
    "blockchain",
    "defi",
    "nft",
    "web3",
    "eth",
    "btc",
    "sol",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class PhraseResult:
    """Result of phrase extraction."""

    phrase: str
    frequency: int
    is_trending: bool

    def to_dict(self) -> dict:
        return {
            "phrase": self.phrase,
            "frequency": self.frequency,
            "is_trending": self.is_trending,
        }


# =============================================================================
# PHRASE CLUSTERER CLASS
# =============================================================================


class PhraseClusterer:
    """
    Extracts and clusters trending phrases from messages.

    Extracts bigrams and trigrams, filters stopwords, and tracks
    frequency to identify trending narratives.

    Requirements: 5.1, 5.2, 5.3, 5.4

    Example:
        >>> clusterer = PhraseClusterer()
        >>> phrases = clusterer.extract_phrases("$MEME to the moon! LFG!")
        >>> print(phrases)  # ['meme moon', 'moon lfg', 'meme moon lfg']
    """

    # Minimum frequency to be considered trending (Requirement 5.3)
    TRENDING_THRESHOLD = 5

    def __init__(
        self,
        stopwords: Optional[set] = None,
        preserve_words: Optional[set] = None,
        min_word_length: int = 2,
        trending_threshold: int = TRENDING_THRESHOLD,
    ):
        """
        Initialize the phrase clusterer.

        Args:
            stopwords: Set of words to filter out (default: STOPWORDS)
            preserve_words: Set of words to always keep (default: CRYPTO_PRESERVE)
            min_word_length: Minimum word length to include (default: 2)
            trending_threshold: Minimum frequency to be trending (default: 5)
        """
        self.stopwords = stopwords or STOPWORDS
        self.preserve_words = preserve_words or CRYPTO_PRESERVE
        self.min_word_length = min_word_length
        self.trending_threshold = trending_threshold

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize and filter text.

        Args:
            text: Input text

        Returns:
            List of filtered tokens
        """
        # Convert to lowercase and extract words
        text = text.lower()
        # Keep alphanumeric, $, #, and common crypto symbols
        words = re.findall(r"[\$#]?\w+", text)

        tokens = []
        for word in words:
            # Remove $ and # prefixes for comparison but keep the word
            clean_word = word.lstrip("$#")

            # Keep if it's a preserved crypto term
            if clean_word in self.preserve_words:
                tokens.append(clean_word)
            # Keep if it's not a stopword and meets length requirement
            elif (
                clean_word not in self.stopwords
                and len(clean_word) >= self.min_word_length
            ):
                tokens.append(clean_word)

        return tokens

    def extract_phrases(self, text: str) -> list[str]:
        """
        Extract bigrams and trigrams from text.

        Args:
            text: Input text to extract phrases from

        Returns:
            List of extracted phrases (bigrams and trigrams)

        Requirements: 5.1
        """
        tokens = self._tokenize(text)
        phrases = []

        # Extract bigrams
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i + 1]}"
            phrases.append(bigram)

        # Extract trigrams
        for i in range(len(tokens) - 2):
            trigram = f"{tokens[i]} {tokens[i + 1]} {tokens[i + 2]}"
            phrases.append(trigram)

        return phrases

    def count_phrases(self, texts: list[str]) -> Counter:
        """
        Count phrase frequencies across multiple texts.

        Args:
            texts: List of text messages

        Returns:
            Counter with phrase frequencies
        """
        all_phrases = []
        for text in texts:
            all_phrases.extend(self.extract_phrases(text))
        return Counter(all_phrases)

    def get_trending_phrases(
        self,
        texts: list[str],
        top_n: int = 10,
    ) -> list[PhraseResult]:
        """
        Get trending phrases from a collection of texts.

        A phrase is trending if it appears >= trending_threshold times.

        Args:
            texts: List of text messages
            top_n: Maximum number of phrases to return

        Returns:
            List of PhraseResult sorted by frequency (descending)

        Requirements: 5.3, 5.4
        """
        phrase_counts = self.count_phrases(texts)

        results = []
        for phrase, count in phrase_counts.most_common():
            is_trending = count >= self.trending_threshold
            results.append(
                PhraseResult(
                    phrase=phrase,
                    frequency=count,
                    is_trending=is_trending,
                )
            )

        # Sort by frequency descending (Requirement 5.4)
        results.sort(key=lambda x: x.frequency, reverse=True)

        return results[:top_n]

    def get_only_trending(
        self,
        texts: list[str],
        top_n: int = 10,
    ) -> list[PhraseResult]:
        """
        Get only phrases that meet the trending threshold.

        Args:
            texts: List of text messages
            top_n: Maximum number of phrases to return

        Returns:
            List of trending PhraseResult sorted by frequency

        Requirements: 5.3
        """
        all_results = self.get_trending_phrases(texts, top_n=100)
        trending = [r for r in all_results if r.is_trending]
        return trending[:top_n]


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

_default_clusterer: Optional[PhraseClusterer] = None


def get_phrase_clusterer() -> PhraseClusterer:
    """Get the default phrase clusterer instance (singleton)."""
    global _default_clusterer
    if _default_clusterer is None:
        _default_clusterer = PhraseClusterer()
    return _default_clusterer


def extract_phrases(text: str) -> list[str]:
    """
    Convenience function to extract phrases using the default clusterer.

    Args:
        text: Input text

    Returns:
        List of extracted phrases
    """
    return get_phrase_clusterer().extract_phrases(text)


def get_trending_phrases(texts: list[str], top_n: int = 10) -> list[PhraseResult]:
    """
    Convenience function to get trending phrases.

    Args:
        texts: List of text messages
        top_n: Maximum number of phrases to return

    Returns:
        List of PhraseResult sorted by frequency
    """
    return get_phrase_clusterer().get_trending_phrases(texts, top_n)
