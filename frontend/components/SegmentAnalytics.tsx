'use client';

import { motion } from 'framer-motion';
import { BarChart3, PieChart, TrendingUp } from 'lucide-react';
import { Segment } from '../lib/types';
import { fadeIn } from '../lib/animations';

interface SegmentAnalyticsProps {
  segments: Segment[];
}

const SEGMENT_COLORS = [
  '#667eea', // Purple
  '#4ade80', // Green
  '#f59e0b', // Amber
  '#ec4899', // Pink
  '#06b6d4', // Cyan
];

export default function SegmentAnalytics({ segments }: SegmentAnalyticsProps) {
  // Calculate analytics
  const totalTranscripts = segments.reduce(
    (acc, seg) => acc + seg.transcripts.length,
    0
  );

  const topicDistribution = segments.reduce((acc, seg, idx) => {
    const topic = seg.topic || 'Unknown';
    if (!acc[topic]) {
      acc[topic] = { count: 0, color: SEGMENT_COLORS[idx % SEGMENT_COLORS.length] };
    }
    acc[topic].count += seg.transcripts.length;
    return acc;
  }, {} as Record<string, { count: number; color: string }>);

  const topicData = Object.entries(topicDistribution)
    .map(([topic, data]) => ({ topic, ...data }))
    .sort((a, b) => b.count - a.count);

  const avgTranscriptsPerSegment =
    segments.length > 0 ? (totalTranscripts / segments.length).toFixed(1) : '0';

  const durations = segments
    .filter((s) => s.completed_at)
    .map((s) => {
      const start = new Date(s.started_at).getTime();
      const end = new Date(s.completed_at!).getTime();
      return (end - start) / 1000; // Convert to seconds
    });

  const avgDuration =
    durations.length > 0
      ? (durations.reduce((a, b) => a + b, 0) / durations.length / 60).toFixed(1)
      : '0';

  return (
    <motion.div {...fadeIn} className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-5">
        <BarChart3 size={18} className="text-[var(--color-success)]" />
        <h2 className="text-lg font-semibold">Analytics</h2>
      </div>

      {segments.length === 0 ? (
        <p className="text-[var(--text-tertiary)] text-sm text-center py-8">
          Analytics will appear as segments are completed.
        </p>
      ) : (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-3 gap-4">
            <div className="bg-[var(--color-surface)] rounded-xl p-4">
              <div className="text-2xl font-bold text-[var(--color-active)]">
                {segments.length}
              </div>
              <div className="text-xs text-[var(--text-tertiary)] mt-1">
                Segments
              </div>
            </div>
            <div className="bg-[var(--color-surface)] rounded-xl p-4">
              <div className="text-2xl font-bold text-[var(--color-success)]">
                {avgTranscriptsPerSegment}
              </div>
              <div className="text-xs text-[var(--text-tertiary)] mt-1">
                Avg Messages
              </div>
            </div>
            <div className="bg-[var(--color-surface)] rounded-xl p-4">
              <div className="text-2xl font-bold text-[var(--color-warning)]">
                {avgDuration}m
              </div>
              <div className="text-xs text-[var(--text-tertiary)] mt-1">
                Avg Duration
              </div>
            </div>
          </div>

          {/* Topic Distribution */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <PieChart size={16} className="text-[var(--text-secondary)]" />
              <h3 className="text-sm font-semibold text-[var(--text-secondary)]">
                Topic Distribution
              </h3>
            </div>
            <div className="space-y-2">
              {topicData.map(({ topic, count, color }) => {
                const percentage =
                  totalTranscripts > 0
                    ? ((count / totalTranscripts) * 100).toFixed(0)
                    : 0;

                return (
                  <div key={topic} className="min-w-0">
                    <div className="flex items-center justify-between text-xs mb-1 gap-2">
                      <span
                        className="font-medium truncate flex-1 min-w-0"
                        style={{ color }}
                      >
                        {topic}
                      </span>
                      <span className="text-[var(--text-tertiary)] flex-shrink-0">
                        {count} ({percentage}%)
                      </span>
                    </div>
                    <div className="h-2 bg-[var(--color-surface)] rounded-full overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${percentage}%` }}
                        transition={{ duration: 0.5, ease: 'easeOut' }}
                        className="h-full rounded-full"
                        style={{ backgroundColor: color }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Segment Duration Chart */}
          {durations.length > 0 && (
            <div>
              <div className="flex items-center gap-2 mb-3">
                <TrendingUp size={16} className="text-[var(--text-secondary)]" />
                <h3 className="text-sm font-semibold text-[var(--text-secondary)]">
                  Segment Durations
                </h3>
              </div>
              <div className="space-y-2">
                {segments.slice(-5).filter((s) => s.completed_at).map((segment, idx) => {
                  const start = new Date(segment.started_at).getTime();
                  const end = new Date(segment.completed_at!).getTime();
                  const duration = (end - start) / 1000 / 60; // minutes
                  const maxDuration = Math.max(...durations) / 60;
                  const widthPercent =
                    maxDuration > 0 ? (duration / maxDuration) * 100 : 0;

                  return (
                    <div key={segment.id} className="min-w-0">
                      <div className="flex items-center justify-between text-xs mb-1 gap-2">
                        <span className="text-[var(--text-secondary)] truncate flex-1 min-w-0">
                          {segment.topic}
                        </span>
                        <span className="text-[var(--text-tertiary)] flex-shrink-0">
                          {duration.toFixed(1)}m
                        </span>
                      </div>
                      <div className="h-2 bg-[var(--color-surface)] rounded-full overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${widthPercent}%` }}
                          transition={{ duration: 0.5, ease: 'easeOut' }}
                          className="h-full rounded-full"
                          style={{
                            backgroundColor:
                              SEGMENT_COLORS[idx % SEGMENT_COLORS.length],
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
