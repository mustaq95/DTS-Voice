'use client';

import { useState, useCallback, useMemo } from 'react';
import { Segment, SegmentUpdate } from '../lib/types';

interface UseSegmentsReturn {
  segments: Segment[];
  activeSegment: Segment | null;
  segmentUpdate: SegmentUpdate | null;
  addCompletedSegment: (segment: Segment) => void;
  updateCurrentSegment: (update: SegmentUpdate) => void;
  getSegmentById: (id: string) => Segment | undefined;
  segmentAnalytics: {
    totalSegments: number;
    totalTranscripts: number;
    averageDuration: number;
    topicDistribution: { topic: string; count: number }[];
  };
}

export function useSegments(): UseSegmentsReturn {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [segmentUpdate, setSegmentUpdate] = useState<SegmentUpdate | null>(null);

  // Add a completed segment to the history
  const addCompletedSegment = useCallback((segment: Segment) => {
    setSegments((prev) => {
      // Check if segment already exists (prevent duplicates)
      const exists = prev.some((s) => s.id === segment.id);
      if (exists) {
        return prev;
      }
      return [...prev, { ...segment, status: 'completed' }];
    });
  }, []);

  // Update the current segment buffer status
  const updateCurrentSegment = useCallback((update: SegmentUpdate) => {
    setSegmentUpdate(update);
  }, []);

  // Get segment by ID
  const getSegmentById = useCallback(
    (id: string) => {
      return segments.find((s) => s.id === id);
    },
    [segments]
  );

  // Find the active segment (most recent or explicitly marked as active)
  const activeSegment = useMemo(() => {
    const activeSegments = segments.filter((s) => s.status === 'active');
    if (activeSegments.length > 0) {
      return activeSegments[activeSegments.length - 1];
    }
    // If no active segments, return the most recent completed one
    if (segments.length > 0) {
      return segments[segments.length - 1];
    }
    return null;
  }, [segments]);

  // Calculate analytics
  const segmentAnalytics = useMemo(() => {
    const totalSegments = segments.length;
    const totalTranscripts = segments.reduce(
      (acc, seg) => acc + seg.transcripts.length,
      0
    );

    // Calculate average duration (in seconds)
    const durations = segments
      .filter((s) => s.completed_at)
      .map((s) => {
        const start = new Date(s.started_at).getTime();
        const end = new Date(s.completed_at!).getTime();
        return (end - start) / 1000; // Convert to seconds
      });

    const averageDuration =
      durations.length > 0
        ? durations.reduce((a, b) => a + b, 0) / durations.length
        : 0;

    // Topic distribution
    const topicCounts = segments.reduce((acc, seg) => {
      const topic = seg.topic || 'Unknown';
      acc[topic] = (acc[topic] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    const topicDistribution = Object.entries(topicCounts).map(
      ([topic, count]) => ({
        topic,
        count,
      })
    );

    return {
      totalSegments,
      totalTranscripts,
      averageDuration,
      topicDistribution,
    };
  }, [segments]);

  return {
    segments,
    activeSegment,
    segmentUpdate,
    addCompletedSegment,
    updateCurrentSegment,
    getSegmentById,
    segmentAnalytics,
  };
}
