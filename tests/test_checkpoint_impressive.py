"""
Checkpoint 16: Impressive Features Complete Verification Tests

This module verifies that Phase 3 (Impressive Features) are complete:
1. Phrase clustering and trending detection work correctly
2. Influencer tracking and consensus calculation work correctly
3. Full pulse score calculation with all components works correctly
4. Performance monitoring is in place

Requirements: Checkpoint 16 verification
"""

import sys
from unittest.mock import MagicMock

# Mock pathway module before importing modules that depend on it
mock_pw = MagicMock()
mock_pw.Schema = type("Schema", (), {})
mock_pw.Duration = MagicMock()
mock_pw.DateTimeUtc = MagicMock()
sys.modules["pathway"] = mock_pw

import pytest

from simulator.hype_simulator import (
    PHASE_ORDER,
    generate_single_message,
)
from transforms.divergence import detect_divergence
from transforms.influence import (
    InfluenceCalculator,
    calculate_consensus,
    calculate_influence_score,
    is_influencer,
)
from transforms.phrase_clusterer import (
    PhraseClusterer,
)
from transforms.pulse_score import (
    PulseScoreCalculator,
)
from transforms.sentiment import SentimentAnalyzer

# =============================================================================
# TEST 1: Phrase Clustering Works
# =============================================================================


class TestPhraseClusteringWorks:
    """Verify phrase clustering extracts and tracks trending phrases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clusterer = PhraseClusterer()

    def test_extracts_bigrams_from_crypto_text(self):
        """Should extract meaningful bigrams from crypto discussions."""
        text = "$MEME to the moon! Diamond hands forever!"
        phrases = self.clusterer.extract_phrases(text)

        # Should have bigrams
        assert len(phrases) > 0
        # Should contain crypto-relevant phrases
        assert any("moon" in p for p in phrases)
        assert any("diamond" in p for p in phrases)

    def test_extracts_trigrams_from_crypto_text(self):
        """Should extract meaningful trigrams from crypto discussions."""
        text = "moon pump bullish hodl diamond hands forever"
        phrases = self.clusterer.extract_phrases(text)

        # Should have trigrams (3-word phrases)
        trigrams = [p for p in phrases if len(p.split()) == 3]
        assert len(trigrams) > 0

    def test_identifies_trending_phrases(self):
        """Should identify phrases that appear frequently as trending."""
        # Simulate messages with repeated phrases
        texts = [
            "$MEME to the moon!",
            "To the moon we go!",
            "Moon soon, diamond hands!",
            "Diamond hands forever!",
            "Diamond hands are key!",
            "Moon moon moon!",
        ]

        results = self.clusterer.get_trending_phrases(texts, top_n=10)

        # Should have results
        assert len(results) > 0

        # Results should be sorted by frequency
        frequencies = [r.frequency for r in results]
        assert frequencies == sorted(frequencies, reverse=True)

    def test_trending_threshold_works(self):
        """Phrases appearing >= 5 times should be marked as trending."""
        # Create texts where "moon pump" appears exactly 5 times
        texts = ["moon pump"] * 5 + ["diamond hands"] * 2

        results = self.clusterer.get_trending_phrases(texts)

        # Find moon pump result
        moon_pump = next((r for r in results if r.phrase == "moon pump"), None)
        assert moon_pump is not None
        assert moon_pump.is_trending is True
        assert moon_pump.frequency == 5

    def test_filters_stopwords_correctly(self):
        """Should filter common stopwords but preserve crypto terms."""
        text = "the moon is going to pump very soon"
        phrases = self.clusterer.extract_phrases(text)

        # Should not have phrases with only stopwords
        for phrase in phrases:
            words = phrase.split()
            # At least one word should be meaningful (not a stopword)
            assert any(w in ["moon", "pump", "soon"] for w in words)


# =============================================================================
# TEST 2: Influencer Tracking Works
# =============================================================================


class TestInfluencerTrackingWorks:
    """Verify influencer tracking identifies and tracks influential accounts."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = InfluenceCalculator()

    def test_influence_score_formula_correct(self):
        """Influence score should be (followers × 0.6) + (engagement × 0.4)."""
        # Test case: 100000 followers, 5000 engagement
        # Expected: 100000 * 0.6 + 5000 * 0.4 = 60000 + 2000 = 62000
        score = calculate_influence_score(followers=100000, engagement=5000)
        assert score == 62000.0

    def test_influencer_classification_threshold(self):
        """Authors with score > 10000 should be classified as influencers."""
        # High influence (should be influencer)
        assert is_influencer(followers=20000, engagement=1000) is True

        # Low influence (should not be influencer)
        assert is_influencer(followers=100, engagement=50) is False

        # Edge case: exactly at threshold
        # 16666 * 0.6 + 1 * 0.4 = 9999.6 + 0.4 = 10000 (not > 10000)
        assert is_influencer(followers=16666, engagement=1) is False

    def test_consensus_calculation_strongly_bullish(self):
        """Should calculate strongly bullish consensus (ratio > 0.7)."""
        # 8 bullish, 2 bearish = 80% bullish
        sentiments = [0.5, 0.6, 0.7, 0.8, 0.4, 0.5, 0.6, 0.7, -0.5, -0.6]
        consensus = calculate_consensus(sentiments)

        assert consensus.bullish_count == 8
        assert consensus.bearish_count == 2
        assert consensus.consensus_ratio == 0.8
        assert consensus.consensus_label == "strongly_bullish"

    def test_consensus_calculation_strongly_bearish(self):
        """Should calculate strongly bearish consensus (ratio < 0.3)."""
        # 2 bullish, 8 bearish = 20% bullish
        sentiments = [0.5, 0.6, -0.5, -0.6, -0.7, -0.8, -0.4, -0.5, -0.6, -0.7]
        consensus = calculate_consensus(sentiments)

        assert consensus.bullish_count == 2
        assert consensus.bearish_count == 8
        assert consensus.consensus_ratio == 0.2
        assert consensus.consensus_label == "strongly_bearish"

    def test_consensus_calculation_neutral(self):
        """Should calculate neutral consensus (ratio around 0.5)."""
        # 5 bullish, 5 bearish = 50% bullish
        sentiments = [0.5, 0.6, 0.7, 0.8, 0.4, -0.5, -0.6, -0.7, -0.8, -0.4]
        consensus = calculate_consensus(sentiments)

        assert consensus.bullish_count == 5
        assert consensus.bearish_count == 5
        assert consensus.consensus_ratio == 0.5
        assert consensus.consensus_label == "neutral"

    def test_filter_influencers_from_messages(self):
        """Should filter only influencer accounts from message list."""
        messages = [
            {
                "author_id": "whale",
                "author_followers": 100000,
                "engagement_count": 5000,
            },
            {"author_id": "regular", "author_followers": 500, "engagement_count": 20},
            {
                "author_id": "medium",
                "author_followers": 20000,
                "engagement_count": 5000,
            },
        ]

        influencers = self.calculator.filter_influencers(messages)

        # Should only include whale and medium (both > 10000 score)
        # whale: 100000 * 0.6 + 5000 * 0.4 = 62000 (influencer)
        # regular: 500 * 0.6 + 20 * 0.4 = 308 (not influencer)
        # medium: 20000 * 0.6 + 5000 * 0.4 = 14000 (influencer)
        assert len(influencers) == 2
        author_ids = [i["author_id"] for i in influencers]
        assert "whale" in author_ids
        assert "medium" in author_ids
        assert "regular" not in author_ids


# =============================================================================
# TEST 3: Full Pulse Score with All Components
# =============================================================================


class TestFullPulseScoreCalculation:
    """Verify pulse score calculation combines all components correctly."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PulseScoreCalculator()

    def test_maximum_score_scenario(self):
        """Maximum inputs should produce score of 10."""
        # High sentiment (>0.7) = 4 points
        # High phrase freq (>20) = 3 points
        # High influencer ratio (>0.7) = 3 points
        # No divergence = 0 penalty
        # Total = 10 (capped)
        score = self.calculator.calculate(
            sentiment_velocity=0.9,
            phrase_frequency=30,
            influencer_ratio=0.9,
            divergence_type="aligned",
        )
        assert score == 10.0

    def test_minimum_score_scenario(self):
        """Minimum inputs should produce score of 1."""
        # Low sentiment = 0 points
        # Low phrase freq = 0 points
        # Low influencer ratio = 0 points
        # Total = 0, clamped to 1
        score = self.calculator.calculate(
            sentiment_velocity=0.1,
            phrase_frequency=3,
            influencer_ratio=0.2,
            divergence_type="aligned",
        )
        assert score == 1.0

    def test_bearish_divergence_penalty(self):
        """Bearish divergence should subtract 1 point."""
        score_aligned = self.calculator.calculate(
            sentiment_velocity=0.8,
            phrase_frequency=25,
            influencer_ratio=0.8,
            divergence_type="aligned",
        )
        score_divergent = self.calculator.calculate(
            sentiment_velocity=0.8,
            phrase_frequency=25,
            influencer_ratio=0.8,
            divergence_type="bearish_divergence",
        )

        assert score_aligned - score_divergent == 1.0

    def test_mid_range_score_calculation(self):
        """Mid-range inputs should produce mid-range score."""
        # Mid sentiment (>0.4) = 2 points
        # Mid phrase freq (>10) = 1.5 points
        # Mid influencer ratio (>0.5) = 1.5 points
        # Total = 5 points
        score = self.calculator.calculate(
            sentiment_velocity=0.5,
            phrase_frequency=15,
            influencer_ratio=0.6,
            divergence_type="aligned",
        )
        assert score == 5.0

    def test_signal_type_classification(self):
        """Signal types should be classified correctly based on score."""
        assert self.calculator.get_signal_type(8.0) == "strong_buy"
        assert self.calculator.get_signal_type(7.0) == "strong_buy"
        assert self.calculator.get_signal_type(3.0) == "cooling_off"
        assert self.calculator.get_signal_type(2.0) == "cooling_off"
        assert self.calculator.get_signal_type(5.0) == "neutral"


# =============================================================================
# TEST 4: Integration - All Components Together
# =============================================================================


class TestFullIntegration:
    """Verify all impressive features work together."""

    def setup_method(self):
        """Set up test fixtures."""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.phrase_clusterer = PhraseClusterer()
        self.influence_calculator = InfluenceCalculator()
        self.pulse_calculator = PulseScoreCalculator()

    def test_end_to_end_impressive_features(self):
        """Test complete flow from messages to pulse score."""
        # Generate simulated messages
        messages = []
        for phase in PHASE_ORDER:
            for _ in range(10):
                msg = generate_single_message(coin_symbol="MEME", phase=phase)
                messages.append(msg)

        # Step 1: Analyze sentiment for all messages
        sentiments = []
        for msg in messages:
            sentiment = self.sentiment_analyzer.analyze(msg["text"])
            sentiments.append(sentiment)

        # Calculate average sentiment (velocity)
        avg_sentiment = sum(sentiments) / len(sentiments)
        assert -1.0 <= avg_sentiment <= 1.0

        # Step 2: Extract and count phrases
        texts = [msg["text"] for msg in messages]
        trending = self.phrase_clusterer.get_trending_phrases(texts, top_n=5)

        # Get max frequency
        max_freq = trending[0].frequency if trending else 0

        # Step 3: Calculate influencer consensus
        influencer_messages = self.influence_calculator.filter_influencers(messages)
        if influencer_messages:
            influencer_sentiments = [
                self.sentiment_analyzer.analyze(msg["text"])
                for msg in influencer_messages
            ]
            consensus = self.influence_calculator.calculate_consensus(
                influencer_sentiments
            )
            influencer_ratio = consensus.consensus_ratio
        else:
            influencer_ratio = 0.5

        # Step 4: Detect divergence (simulated price data)
        price_delta = 2.0  # Simulated 2% price increase
        divergence = detect_divergence(avg_sentiment, price_delta)

        # Step 5: Calculate final pulse score
        score = self.pulse_calculator.calculate(
            sentiment_velocity=avg_sentiment,
            phrase_frequency=max_freq,
            influencer_ratio=influencer_ratio,
            divergence_type=divergence,
        )

        # Verify score is valid
        assert 1.0 <= score <= 10.0

        # Verify signal type
        signal = self.pulse_calculator.get_signal_type(score)
        assert signal in ["strong_buy", "cooling_off", "neutral"]

    def test_phrase_clustering_with_simulator_messages(self):
        """Phrase clustering should work with simulator-generated messages."""
        # Generate peak phase messages (high sentiment, trending phrases)
        messages = []
        for _ in range(20):
            msg = generate_single_message(coin_symbol="MEME", phase="peak")
            messages.append(msg)

        texts = [msg["text"] for msg in messages]
        trending = self.phrase_clusterer.get_trending_phrases(texts, top_n=10)

        # Should find some phrases
        assert len(trending) > 0

        # Top phrases should have reasonable frequency
        if trending:
            assert trending[0].frequency >= 1

    def test_influencer_tracking_with_simulator_messages(self):
        """Influencer tracking should work with simulator-generated messages."""
        # Generate messages (some will be from influencers)
        messages = []
        for phase in PHASE_ORDER:
            for _ in range(25):
                msg = generate_single_message(coin_symbol="MEME", phase=phase)
                messages.append(msg)

        # Filter influencers
        influencers = self.influence_calculator.filter_influencers(messages)

        # Should have some influencers (simulator includes them)
        # Note: Due to randomness, we just check the filtering works
        assert isinstance(influencers, list)

        # If we have influencers, calculate consensus
        if influencers:
            sentiments = [
                self.sentiment_analyzer.analyze(msg["text"]) for msg in influencers
            ]
            consensus = self.influence_calculator.calculate_consensus(sentiments)

            assert 0.0 <= consensus.consensus_ratio <= 1.0
            assert consensus.consensus_label in [
                "strongly_bullish",
                "moderately_bullish",
                "neutral",
                "moderately_bearish",
                "strongly_bearish",
            ]


# =============================================================================
# TEST 5: Performance Monitoring Components
# =============================================================================


class TestPerformanceMonitoring:
    """Verify performance monitoring components are in place."""

    def test_performance_module_exists(self):
        """Performance module should exist and be importable."""
        from transforms.performance import (
            PerformanceMonitor,
            get_performance_monitor,
        )

        monitor = get_performance_monitor()
        assert monitor is not None
        assert isinstance(monitor, PerformanceMonitor)

    def test_latency_tracking(self):
        """Should track end-to-end latency."""
        from transforms.performance import (
            get_performance_monitor,
            reset_performance_monitor,
        )

        reset_performance_monitor()
        monitor = get_performance_monitor()

        # Record ingestion and alert
        monitor.record_ingestion("msg_1", ingestion_time=1000.0)
        monitor.record_alert("msg_1", alert_time=1000.5)  # 500ms latency

        monitor.record_ingestion("msg_2", ingestion_time=1001.0)
        monitor.record_alert("msg_2", alert_time=1002.0)  # 1000ms latency

        metrics = monitor.get_metrics()

        assert metrics.avg_latency_ms > 0
        assert metrics.max_latency_ms >= metrics.avg_latency_ms

    def test_throughput_tracking(self):
        """Should track message throughput."""
        from transforms.performance import (
            get_performance_monitor,
            reset_performance_monitor,
        )

        reset_performance_monitor()
        monitor = get_performance_monitor()

        # Record some messages
        monitor.record_message_processed()
        monitor.record_message_processed()
        monitor.record_message_processed()

        metrics = monitor.get_metrics()

        assert metrics.total_messages == 3

    def test_latency_warning_threshold(self):
        """Should warn when latency exceeds 5 seconds."""
        from transforms.performance import (
            LatencyTracker,
        )

        tracker = LatencyTracker(warning_threshold_seconds=5.0)

        # Record ingestion
        tracker.record_ingestion("msg_high", ingestion_time=1000.0)

        # Record alert with high latency (6 seconds)
        latency = tracker.record_alert("msg_high", alert_time=1006.0)

        # Should have recorded the latency
        assert latency is not None
        assert latency == 6000.0  # 6000ms

        # Check statistics show warning
        stats = tracker.get_statistics()
        assert stats["warnings_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
