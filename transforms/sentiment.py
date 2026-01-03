"""
Sentiment Analysis module for the Crypto Narrative Pulse Tracker.

Provides VADER-based sentiment analysis with crypto-specific lexicon
for analyzing social media messages about cryptocurrencies.

Requirements: 3.1
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SentimentAnalyzer:
    """
    Analyzes sentiment using VADER with crypto-specific lexicon.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is enhanced
    with crypto-specific terms to better capture sentiment in crypto
    social media discussions.

    The analyzer returns a compound score in the range [-1, 1]:
    - -1: Most negative sentiment
    - 0: Neutral sentiment
    - 1: Most positive sentiment

    Requirements: 3.1

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> analyzer.analyze("$MEME to the moon! 🚀")
        0.8  # Highly positive
        >>> analyzer.analyze("This is a rug pull scam")
        -0.9  # Highly negative
    """

    # Crypto-specific lexicon with sentiment scores
    # Positive scores indicate bullish sentiment, negative indicate bearish
    CRYPTO_LEXICON = {
        # Bullish terms
        "moon": 3.0,
        "mooning": 3.5,
        "pump": 2.5,
        "pumping": 2.5,
        "bullish": 2.5,
        "hodl": 1.5,
        "hodling": 1.5,
        "fomo": 1.0,  # Fear of missing out - slightly positive
        "degen": 0.5,  # Degenerate trader - neutral to slightly positive in crypto
        "lfg": 2.5,  # Let's f***ing go - very bullish
        "wagmi": 2.0,  # We're all gonna make it - bullish
        "gm": 0.5,  # Good morning - positive community signal
        "alpha": 1.5,  # Insider info - positive
        "gem": 2.0,  # Hidden gem - bullish
        "based": 1.5,  # Positive slang
        "ape": 1.0,  # Aping in - bullish action
        "aped": 1.0,
        "diamond": 2.0,  # Diamond hands - strong holder
        "lambo": 2.5,  # Lamborghini - wealth expectation
        # Bearish terms
        "dump": -3.0,
        "dumping": -3.0,
        "rug": -4.0,  # Rug pull - very negative
        "rugged": -4.0,
        "scam": -3.5,
        "bearish": -2.5,
        "fud": -2.0,  # Fear, uncertainty, doubt - negative
        "rekt": -3.0,  # Wrecked - lost money
        "ngmi": -2.0,  # Not gonna make it - bearish
        "crash": -3.0,
        "crashing": -3.0,
        "ponzi": -3.5,
        "exit": -1.5,  # Exit scam context
        "sell": -1.0,
        "selling": -1.0,
        "paper": -1.5,  # Paper hands - weak holder
        "dead": -2.5,
        "dying": -2.5,
    }

    def __init__(self):
        """Initialize the sentiment analyzer with crypto lexicon."""
        self._analyzer = SentimentIntensityAnalyzer()
        self._add_crypto_lexicon()

    def _add_crypto_lexicon(self) -> None:
        """Add crypto-specific terms to VADER's lexicon."""
        self._analyzer.lexicon.update(self.CRYPTO_LEXICON)

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of the given text.

        Args:
            text: The text to analyze (e.g., social media message)

        Returns:
            Compound sentiment score in range [-1, 1]:
            - Positive values indicate bullish sentiment
            - Negative values indicate bearish sentiment
            - Values near 0 indicate neutral sentiment

        Requirements: 3.1
        """
        if not text or not text.strip():
            return 0.0

        scores = self._analyzer.polarity_scores(text)
        return scores["compound"]

    def analyze_detailed(self, text: str) -> dict:
        """
        Get detailed sentiment breakdown.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with keys:
            - neg: Negative sentiment proportion (0-1)
            - neu: Neutral sentiment proportion (0-1)
            - pos: Positive sentiment proportion (0-1)
            - compound: Overall sentiment score (-1 to 1)
        """
        if not text or not text.strip():
            return {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0}

        return self._analyzer.polarity_scores(text)

    def classify_momentum(self, sentiment_velocity: float) -> str:
        """
        Classify sentiment velocity into momentum categories.

        Args:
            sentiment_velocity: Average sentiment over a time window

        Returns:
            Momentum classification string:
            - "strong_bullish_momentum" if velocity > 0.7
            - "strong_bearish_momentum" if velocity < -0.7
            - "moderate_bullish" if velocity > 0.3
            - "moderate_bearish" if velocity < -0.3
            - "neutral" otherwise

        Requirements: 3.3, 3.4
        """
        if sentiment_velocity > 0.7:
            return "strong_bullish_momentum"
        elif sentiment_velocity < -0.7:
            return "strong_bearish_momentum"
        elif sentiment_velocity > 0.3:
            return "moderate_bullish"
        elif sentiment_velocity < -0.3:
            return "moderate_bearish"
        else:
            return "neutral"


# Module-level singleton for convenience
_default_analyzer: SentimentAnalyzer | None = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """
    Get the default sentiment analyzer instance (singleton).

    Returns:
        SentimentAnalyzer instance
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = SentimentAnalyzer()
    return _default_analyzer


def analyze_sentiment(text: str) -> float:
    """
    Convenience function to analyze sentiment using the default analyzer.

    Args:
        text: The text to analyze

    Returns:
        Compound sentiment score in range [-1, 1]
    """
    return get_sentiment_analyzer().analyze(text)
