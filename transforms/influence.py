"""
Influence Calculator module for the Crypto Narrative Pulse Tracker.

Calculates influence scores for message authors and tracks influencer
sentiment to determine market consensus.

Requirements: 6.1, 6.2, 6.3, 6.4
"""

from dataclasses import dataclass
from typing import Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

# Influence score threshold for "influencer" classification (Requirement 6.2)
INFLUENCER_THRESHOLD = 10000

# Sentiment thresholds for bullish/bearish classification
BULLISH_SENTIMENT_THRESHOLD = 0.3
BEARISH_SENTIMENT_THRESHOLD = -0.3


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class InfluenceResult:
    """Result of influence calculation for an author."""

    author_id: str
    influence_score: float
    is_influencer: bool
    followers: int
    engagement: int

    def to_dict(self) -> dict:
        return {
            "author_id": self.author_id,
            "influence_score": self.influence_score,
            "is_influencer": self.is_influencer,
            "followers": self.followers,
            "engagement": self.engagement,
        }


@dataclass
class InfluencerConsensus:
    """Aggregated influencer sentiment consensus."""

    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_count: int
    consensus_ratio: float  # bullish / total
    consensus_label: str  # "strongly_bullish", "moderately_bullish", etc.

    def to_dict(self) -> dict:
        return {
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "total_count": self.total_count,
            "consensus_ratio": self.consensus_ratio,
            "consensus_label": self.consensus_label,
        }


# =============================================================================
# INFLUENCE CALCULATOR CLASS
# =============================================================================


class InfluenceCalculator:
    """
    Calculates influence scores and tracks influencer sentiment.

    Influence score formula: (followers × 0.6) + (engagement × 0.4)

    Requirements: 6.1, 6.2, 6.3, 6.4

    Example:
        >>> calc = InfluenceCalculator()
        >>> score = calc.calculate_score(followers=100000, engagement=5000)
        >>> print(score)  # 62000.0
        >>> print(calc.is_influencer(score))  # True
    """

    # Weight factors for influence calculation (Requirement 6.1)
    FOLLOWER_WEIGHT = 0.6
    ENGAGEMENT_WEIGHT = 0.4

    def __init__(
        self,
        influencer_threshold: float = INFLUENCER_THRESHOLD,
        bullish_threshold: float = BULLISH_SENTIMENT_THRESHOLD,
        bearish_threshold: float = BEARISH_SENTIMENT_THRESHOLD,
    ):
        """
        Initialize the influence calculator.

        Args:
            influencer_threshold: Score threshold for influencer status
            bullish_threshold: Sentiment threshold for bullish classification
            bearish_threshold: Sentiment threshold for bearish classification
        """
        self.influencer_threshold = influencer_threshold
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def calculate_score(self, followers: int, engagement: int) -> float:
        """
        Calculate influence score.

        Formula: (followers × 0.6) + (engagement × 0.4)

        Args:
            followers: Number of followers/subscribers
            engagement: Engagement count (likes, reactions, etc.)

        Returns:
            Influence score

        Requirements: 6.1
        """
        return (followers * self.FOLLOWER_WEIGHT) + (
            engagement * self.ENGAGEMENT_WEIGHT
        )

    def is_influencer(self, score: float) -> bool:
        """
        Check if a score qualifies as an influencer.

        Args:
            score: Influence score

        Returns:
            True if score > threshold

        Requirements: 6.2
        """
        return score > self.influencer_threshold

    def analyze_author(
        self,
        author_id: str,
        followers: int,
        engagement: int,
    ) -> InfluenceResult:
        """
        Analyze an author's influence.

        Args:
            author_id: Author identifier
            followers: Number of followers
            engagement: Engagement count

        Returns:
            InfluenceResult with score and classification
        """
        score = self.calculate_score(followers, engagement)
        return InfluenceResult(
            author_id=author_id,
            influence_score=score,
            is_influencer=self.is_influencer(score),
            followers=followers,
            engagement=engagement,
        )

    def classify_sentiment(self, sentiment: float) -> str:
        """
        Classify sentiment as bullish, bearish, or neutral.

        Args:
            sentiment: Sentiment score (-1 to 1)

        Returns:
            Classification string
        """
        if sentiment > self.bullish_threshold:
            return "bullish"
        elif sentiment < self.bearish_threshold:
            return "bearish"
        return "neutral"

    def calculate_consensus(
        self,
        influencer_sentiments: list[float],
    ) -> InfluencerConsensus:
        """
        Calculate influencer consensus from sentiment scores.

        Args:
            influencer_sentiments: List of sentiment scores from influencers

        Returns:
            InfluencerConsensus with counts and ratio

        Requirements: 6.4
        """
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for sentiment in influencer_sentiments:
            classification = self.classify_sentiment(sentiment)
            if classification == "bullish":
                bullish_count += 1
            elif classification == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1

        total_count = len(influencer_sentiments)

        # Calculate consensus ratio (Requirement 6.4)
        if total_count > 0:
            consensus_ratio = bullish_count / total_count
        else:
            consensus_ratio = 0.5  # Neutral if no data

        # Determine consensus label
        if consensus_ratio > 0.7:
            consensus_label = "strongly_bullish"
        elif consensus_ratio > 0.5:
            consensus_label = "moderately_bullish"
        elif consensus_ratio < 0.3:
            consensus_label = "strongly_bearish"
        elif consensus_ratio < 0.5:
            consensus_label = "moderately_bearish"
        else:
            consensus_label = "neutral"

        return InfluencerConsensus(
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            total_count=total_count,
            consensus_ratio=consensus_ratio,
            consensus_label=consensus_label,
        )

    def filter_influencers(
        self,
        authors: list[dict],
    ) -> list[dict]:
        """
        Filter a list of authors to only influencers.

        Args:
            authors: List of author dicts with 'author_followers' and 'engagement_count'

        Returns:
            List of authors who are influencers

        Requirements: 6.2
        """
        influencers = []
        for author in authors:
            followers = author.get("author_followers", 0)
            engagement = author.get("engagement_count", 0)
            score = self.calculate_score(followers, engagement)
            if self.is_influencer(score):
                influencers.append(author)
        return influencers


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================

_default_calculator: Optional[InfluenceCalculator] = None


def get_influence_calculator() -> InfluenceCalculator:
    """Get the default influence calculator instance (singleton)."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = InfluenceCalculator()
    return _default_calculator


def calculate_influence_score(followers: int, engagement: int) -> float:
    """
    Convenience function to calculate influence score.

    Formula: (followers × 0.6) + (engagement × 0.4)

    Args:
        followers: Number of followers
        engagement: Engagement count

    Returns:
        Influence score

    Requirements: 6.1
    """
    return get_influence_calculator().calculate_score(followers, engagement)


def is_influencer(followers: int, engagement: int) -> bool:
    """
    Check if an author is an influencer based on followers and engagement.

    Args:
        followers: Number of followers
        engagement: Engagement count

    Returns:
        True if influence score > 10000

    Requirements: 6.2
    """
    calc = get_influence_calculator()
    score = calc.calculate_score(followers, engagement)
    return calc.is_influencer(score)


def calculate_consensus(influencer_sentiments: list[float]) -> InfluencerConsensus:
    """
    Calculate influencer consensus from sentiment scores.

    Args:
        influencer_sentiments: List of sentiment scores from influencers

    Returns:
        InfluencerConsensus with counts and ratio

    Requirements: 6.4
    """
    return get_influence_calculator().calculate_consensus(influencer_sentiments)
