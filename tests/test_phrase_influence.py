"""
Tests for Phrase Clusterer and Influence Calculator modules.

Tests the new transforms added for Phase 3:
- transforms/phrase_clusterer.py (Requirements 5.1, 5.2, 5.3, 5.4)
- transforms/influence.py (Requirements 6.1, 6.2, 6.3, 6.4)
"""

from transforms.influence import (
    InfluenceCalculator,
    InfluencerConsensus,
    InfluenceResult,
    calculate_consensus,
    calculate_influence_score,
    get_influence_calculator,
    is_influencer,
)
from transforms.phrase_clusterer import (
    PhraseClusterer,
    PhraseResult,
    extract_phrases,
    get_phrase_clusterer,
    get_trending_phrases,
)

# =============================================================================
# PHRASE CLUSTERER TESTS
# =============================================================================


class TestPhraseClusterer:
    """Tests for PhraseClusterer class."""

    def test_extract_bigrams(self):
        """Test bigram extraction from text."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("moon pump bullish")

        # Should contain bigrams
        assert "moon pump" in phrases
        assert "pump bullish" in phrases

    def test_extract_trigrams(self):
        """Test trigram extraction from text."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("moon pump bullish hodl")

        # Should contain trigrams
        assert "moon pump bullish" in phrases
        assert "pump bullish hodl" in phrases

    def test_filters_stopwords(self):
        """Test that common stopwords are filtered."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("the moon is going to pump")

        # Stopwords should be filtered
        # "the", "is", "going", "to" are stopwords
        # Should get phrases from remaining words
        assert len(phrases) > 0
        # "the moon" should not be a phrase since "the" is filtered
        assert "the moon" not in phrases

    def test_preserves_crypto_terms(self):
        """Test that crypto-specific terms are preserved."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("hodl diamond hands lfg")

        # Crypto terms should be preserved
        assert "hodl diamond" in phrases
        assert "diamond hands" in phrases

    def test_empty_text_returns_empty(self):
        """Test that empty text returns empty list."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("")
        assert phrases == []

    def test_single_word_returns_empty(self):
        """Test that single word returns empty list (no n-grams possible)."""
        clusterer = PhraseClusterer()
        phrases = clusterer.extract_phrases("moon")
        assert phrases == []

    def test_count_phrases(self):
        """Test phrase counting across multiple texts."""
        clusterer = PhraseClusterer()
        texts = [
            "moon pump",
            "moon pump bullish",
            "moon pump lfg",
        ]
        counts = clusterer.count_phrases(texts)

        # "moon pump" appears in all 3 texts
        assert counts["moon pump"] == 3

    def test_trending_threshold(self):
        """Test that trending threshold works correctly."""
        clusterer = PhraseClusterer(trending_threshold=3)
        texts = [
            "moon pump",
            "moon pump",
            "moon pump",  # 3 times - should be trending
            "diamond hands",  # 1 time - not trending
        ]
        results = clusterer.get_trending_phrases(texts)

        # Find moon pump result
        moon_pump = next((r for r in results if r.phrase == "moon pump"), None)
        assert moon_pump is not None
        assert moon_pump.is_trending is True
        assert moon_pump.frequency == 3

    def test_get_only_trending(self):
        """Test filtering to only trending phrases."""
        clusterer = PhraseClusterer(trending_threshold=2)
        texts = [
            "moon pump",
            "moon pump",  # 2 times - trending
            "diamond hands",  # 1 time - not trending
        ]
        trending = clusterer.get_only_trending(texts)

        # Only moon pump should be in trending
        phrases = [r.phrase for r in trending]
        assert "moon pump" in phrases

    def test_results_sorted_by_frequency(self):
        """Test that results are sorted by frequency descending."""
        clusterer = PhraseClusterer()
        texts = [
            "moon pump",
            "moon pump",
            "moon pump",
            "diamond hands",
            "diamond hands",
            "lfg wagmi",
        ]
        results = clusterer.get_trending_phrases(texts)

        # Should be sorted by frequency
        frequencies = [r.frequency for r in results]
        assert frequencies == sorted(frequencies, reverse=True)


class TestPhraseClustererModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_phrase_clusterer_singleton(self):
        """Test that get_phrase_clusterer returns singleton."""
        c1 = get_phrase_clusterer()
        c2 = get_phrase_clusterer()
        assert c1 is c2

    def test_extract_phrases_function(self):
        """Test extract_phrases convenience function."""
        phrases = extract_phrases("moon pump bullish")
        assert "moon pump" in phrases
        assert "pump bullish" in phrases

    def test_get_trending_phrases_function(self):
        """Test get_trending_phrases convenience function."""
        texts = ["moon pump"] * 5 + ["diamond hands"]
        results = get_trending_phrases(texts)
        assert len(results) > 0


# =============================================================================
# INFLUENCE CALCULATOR TESTS
# =============================================================================


class TestInfluenceCalculator:
    """Tests for InfluenceCalculator class."""

    def test_calculate_score_formula(self):
        """Test influence score formula: (followers × 0.6) + (engagement × 0.4)."""
        calc = InfluenceCalculator()

        # 100000 * 0.6 + 5000 * 0.4 = 60000 + 2000 = 62000
        score = calc.calculate_score(followers=100000, engagement=5000)
        assert score == 62000.0

    def test_calculate_score_zero_values(self):
        """Test score calculation with zero values."""
        calc = InfluenceCalculator()
        score = calc.calculate_score(followers=0, engagement=0)
        assert score == 0.0

    def test_is_influencer_above_threshold(self):
        """Test influencer classification above threshold."""
        calc = InfluenceCalculator(influencer_threshold=10000)

        # Score > 10000 should be influencer
        assert calc.is_influencer(15000) is True
        assert calc.is_influencer(10001) is True

    def test_is_influencer_below_threshold(self):
        """Test influencer classification below threshold."""
        calc = InfluenceCalculator(influencer_threshold=10000)

        # Score <= 10000 should not be influencer
        assert calc.is_influencer(10000) is False
        assert calc.is_influencer(5000) is False

    def test_analyze_author(self):
        """Test full author analysis."""
        calc = InfluenceCalculator()
        result = calc.analyze_author(
            author_id="whale_1",
            followers=100000,
            engagement=5000,
        )

        assert isinstance(result, InfluenceResult)
        assert result.author_id == "whale_1"
        assert result.influence_score == 62000.0
        assert result.is_influencer is True
        assert result.followers == 100000
        assert result.engagement == 5000

    def test_classify_sentiment_bullish(self):
        """Test bullish sentiment classification."""
        calc = InfluenceCalculator(bullish_threshold=0.3)
        assert calc.classify_sentiment(0.5) == "bullish"
        assert calc.classify_sentiment(0.31) == "bullish"

    def test_classify_sentiment_bearish(self):
        """Test bearish sentiment classification."""
        calc = InfluenceCalculator(bearish_threshold=-0.3)
        assert calc.classify_sentiment(-0.5) == "bearish"
        assert calc.classify_sentiment(-0.31) == "bearish"

    def test_classify_sentiment_neutral(self):
        """Test neutral sentiment classification."""
        calc = InfluenceCalculator(bullish_threshold=0.3, bearish_threshold=-0.3)
        assert calc.classify_sentiment(0.0) == "neutral"
        assert calc.classify_sentiment(0.2) == "neutral"
        assert calc.classify_sentiment(-0.2) == "neutral"

    def test_calculate_consensus_strongly_bullish(self):
        """Test strongly bullish consensus (ratio > 0.7)."""
        calc = InfluenceCalculator()
        sentiments = [0.5, 0.6, 0.7, 0.8, 0.4]  # 5 bullish, 0 bearish
        consensus = calc.calculate_consensus(sentiments)

        assert consensus.bullish_count == 5
        assert consensus.bearish_count == 0
        assert consensus.consensus_ratio == 1.0
        assert consensus.consensus_label == "strongly_bullish"

    def test_calculate_consensus_strongly_bearish(self):
        """Test strongly bearish consensus (ratio < 0.3)."""
        calc = InfluenceCalculator()
        sentiments = [-0.5, -0.6, -0.7, -0.8, -0.4]  # 0 bullish, 5 bearish
        consensus = calc.calculate_consensus(sentiments)

        assert consensus.bullish_count == 0
        assert consensus.bearish_count == 5
        assert consensus.consensus_ratio == 0.0
        assert consensus.consensus_label == "strongly_bearish"

    def test_calculate_consensus_neutral(self):
        """Test neutral consensus (ratio around 0.5)."""
        calc = InfluenceCalculator()
        sentiments = [0.5, 0.5, -0.5, -0.5]  # 2 bullish, 2 bearish
        consensus = calc.calculate_consensus(sentiments)

        assert consensus.bullish_count == 2
        assert consensus.bearish_count == 2
        assert consensus.consensus_ratio == 0.5
        assert consensus.consensus_label == "neutral"

    def test_calculate_consensus_empty(self):
        """Test consensus with empty list."""
        calc = InfluenceCalculator()
        consensus = calc.calculate_consensus([])

        assert consensus.total_count == 0
        assert consensus.consensus_ratio == 0.5  # Default neutral

    def test_filter_influencers(self):
        """Test filtering authors to only influencers."""
        calc = InfluenceCalculator(influencer_threshold=10000)
        authors = [
            {
                "author_id": "whale",
                "author_followers": 100000,
                "engagement_count": 5000,
            },
            {"author_id": "regular", "author_followers": 100, "engagement_count": 10},
        ]
        influencers = calc.filter_influencers(authors)

        assert len(influencers) == 1
        assert influencers[0]["author_id"] == "whale"


class TestInfluenceModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_influence_calculator_singleton(self):
        """Test that get_influence_calculator returns singleton."""
        c1 = get_influence_calculator()
        c2 = get_influence_calculator()
        assert c1 is c2

    def test_calculate_influence_score_function(self):
        """Test calculate_influence_score convenience function."""
        score = calculate_influence_score(followers=100000, engagement=5000)
        assert score == 62000.0

    def test_is_influencer_function(self):
        """Test is_influencer convenience function."""
        # High followers + engagement = influencer
        assert is_influencer(followers=100000, engagement=5000) is True
        # Low followers + engagement = not influencer
        assert is_influencer(followers=100, engagement=10) is False

    def test_calculate_consensus_function(self):
        """Test calculate_consensus convenience function."""
        sentiments = [0.5, 0.6, -0.5]  # 2 bullish, 1 bearish
        consensus = calculate_consensus(sentiments)

        assert isinstance(consensus, InfluencerConsensus)
        assert consensus.bullish_count == 2
        assert consensus.bearish_count == 1


# =============================================================================
# DATA CLASS TESTS
# =============================================================================


class TestDataClasses:
    """Tests for data class serialization."""

    def test_phrase_result_to_dict(self):
        """Test PhraseResult.to_dict()."""
        result = PhraseResult(phrase="moon pump", frequency=5, is_trending=True)
        d = result.to_dict()

        assert d["phrase"] == "moon pump"
        assert d["frequency"] == 5
        assert d["is_trending"] is True

    def test_influence_result_to_dict(self):
        """Test InfluenceResult.to_dict()."""
        result = InfluenceResult(
            author_id="whale",
            influence_score=62000.0,
            is_influencer=True,
            followers=100000,
            engagement=5000,
        )
        d = result.to_dict()

        assert d["author_id"] == "whale"
        assert d["influence_score"] == 62000.0
        assert d["is_influencer"] is True

    def test_influencer_consensus_to_dict(self):
        """Test InfluencerConsensus.to_dict()."""
        consensus = InfluencerConsensus(
            bullish_count=3,
            bearish_count=1,
            neutral_count=1,
            total_count=5,
            consensus_ratio=0.6,
            consensus_label="moderately_bullish",
        )
        d = consensus.to_dict()

        assert d["bullish_count"] == 3
        assert d["consensus_ratio"] == 0.6
        assert d["consensus_label"] == "moderately_bullish"
