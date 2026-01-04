"use client";

import { useState, useEffect, useCallback } from "react";

export interface PerformanceMetrics {
  latency: {
    avg_ms: number;
    min_ms: number;
    max_ms: number;
    p95_ms: number;
    p99_ms: number;
    warnings_count: number;
  };
  throughput: {
    current_mps: number;
    total_messages: number;
    overall_avg_mps: number;
  };
  timestamp: string;
  uptime_seconds: number;
}

const DEFAULT_METRICS: PerformanceMetrics = {
  latency: {
    avg_ms: 0,
    min_ms: 0,
    max_ms: 0,
    p95_ms: 0,
    p99_ms: 0,
    warnings_count: 0,
  },
  throughput: {
    current_mps: 0,
    total_messages: 0,
    overall_avg_mps: 0,
  },
  timestamp: new Date().toISOString(),
  uptime_seconds: 0,
};

export const usePerformanceData = (
  apiUrl: string = "http://localhost:8000",
  pollInterval: number = 5000
) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>(DEFAULT_METRICS);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  const fetchMetrics = useCallback(async () => {
    try {
      const response = await fetch(`${apiUrl}/api/performance`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setMetrics(data);
      setIsConnected(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch");
      setIsConnected(false);
      // Keep last known metrics on error
    } finally {
      setIsLoading(false);
    }
  }, [apiUrl]);

  useEffect(() => {
    // Initial fetch
    fetchMetrics();

    // Set up polling
    const interval = setInterval(fetchMetrics, pollInterval);

    return () => clearInterval(interval);
  }, [fetchMetrics, pollInterval]);

  return {
    metrics,
    isLoading,
    error,
    isConnected,
    refetch: fetchMetrics,
  };
};

// Simulated performance data for demo mode (when API is not available)
export const useSimulatedPerformanceData = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics>(DEFAULT_METRICS);

  useEffect(() => {
    let messageCount = 0;
    const startTime = Date.now();

    const interval = setInterval(() => {
      messageCount += Math.floor(Math.random() * 5) + 1;
      const uptime = (Date.now() - startTime) / 1000;

      setMetrics({
        latency: {
          avg_ms: 50 + Math.random() * 100,
          min_ms: 20 + Math.random() * 30,
          max_ms: 100 + Math.random() * 200,
          p95_ms: 80 + Math.random() * 150,
          p99_ms: 150 + Math.random() * 200,
          warnings_count: Math.random() > 0.95 ? 1 : 0,
        },
        throughput: {
          current_mps: 2 + Math.random() * 8,
          total_messages: messageCount,
          overall_avg_mps: messageCount / Math.max(uptime, 1),
        },
        timestamp: new Date().toISOString(),
        uptime_seconds: uptime,
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return {
    metrics,
    isLoading: false,
    error: null,
    isConnected: true,
    refetch: () => {},
  };
};
