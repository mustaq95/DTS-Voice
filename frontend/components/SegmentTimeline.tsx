'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, MessageSquare } from 'lucide-react';
import { Segment, SegmentUpdate, Transcript } from '../lib/types';
import { useLiveDuration } from '../hooks/useLiveDuration';

interface SegmentTimelineProps {
  segments: Segment[];
  currentSegment?: SegmentUpdate | null;
  transcripts?: Transcript[];
  onSegmentClick?: (segmentId: string) => void;
}

const SEGMENT_COLORS = [
  'var(--segment-1)', // Purple
  'var(--segment-2)', // Green
  'var(--segment-3)', // Amber
  'var(--segment-4)', // Pink
  'var(--segment-5)', // Cyan
];

function formatDuration(startedAt: string, completedAt?: string): string {
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const durationMs = end - start;
  const minutes = Math.floor(durationMs / 60000);
  const seconds = Math.floor((durationMs % 60000) / 1000);

  if (minutes > 0) {
    return `${minutes}m ${seconds}s`;
  }
  return `${seconds}s`;
}

function getActionColor(action: string): { backgroundColor: string; color: string } {
  switch (action) {
    case 'NEW_TOPIC':
      return {
        backgroundColor: 'rgba(239, 68, 68, 0.2)',
        color: '#ef4444'
      };
    case 'CONTINUE':
      return {
        backgroundColor: 'rgba(74, 222, 128, 0.15)',
        color: '#4ade80'
      };
    case 'NOISE':
      return {
        backgroundColor: 'rgba(251, 191, 36, 0.2)',
        color: '#fbbf24'
      };
    default:
      return {
        backgroundColor: 'rgba(59, 130, 246, 0.2)',
        color: '#3b82f6'
      };
  }
}

export default function SegmentTimeline({
  segments,
  currentSegment,
  transcripts = [],
  onSegmentClick,
}: SegmentTimelineProps) {
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);
  const liveDuration = useLiveDuration(currentSegment?.segment_started_at);

  // Build combined list: completed segments + current active segment (if exists)
  const allSegments = React.useMemo(() => {
    const combined = [...segments];

    // Add current segment as last item if it exists AND is not already in segments array
    if (currentSegment && currentSegment.classification_status) {
      // Check if an active segment already exists in segments array (from segment_active message)
      const hasActiveSegment = segments.some(seg => seg.status === 'active');

      // Only add currentSegment if there's NO active segment in the segments array
      // This prevents duplication since both segment_active and buffer_update messages create the same segment
      if (!hasActiveSegment) {
        combined.push({
          id: 'active-segment',
          topic: currentSegment.current_topic || 'Unknown Topic',
          transcripts: [], // Active segment transcripts not needed for display
          status: 'active',
          started_at: currentSegment.segment_started_at || new Date().toISOString(),
          completed_at: undefined,
        });
      }
    }

    return combined;
  }, [segments, currentSegment]);

  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-5">
        <Clock size={18} className="text-[var(--color-active)]" />
        <h2 className="text-lg font-semibold">Segment Timeline</h2>
      </div>

      {allSegments.length === 0 && (
        <p className="text-[var(--text-tertiary)] text-sm text-center py-8">
          No segments yet. Start speaking to see topics appear.
        </p>
      )}

      <div className="space-y-3">
        {allSegments.map((segment, index) => {
          const isActive = segment.status === 'active';
          const segmentColor = SEGMENT_COLORS[index % SEGMENT_COLORS.length];
          const duration = isActive
            ? liveDuration
            : formatDuration(segment.started_at, segment.completed_at);

          return (
            <motion.div
              key={segment.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => !isActive && onSegmentClick?.(segment.id)}
              onMouseEnter={() => setHoveredSegment(segment.id)}
              onMouseLeave={() => setHoveredSegment(null)}
              className={`relative p-4 rounded-xl transition-all ${
                isActive
                  ? 'border-2'
                  : `border ${onSegmentClick ? 'cursor-pointer hover:bg-[var(--color-surface)] hover:border-opacity-30 hover:scale-[1.02]' : ''}`
              }`}
              style={
                isActive
                  ? {
                      backgroundColor: 'rgba(74, 222, 128, 0.1)',
                      borderColor: 'rgb(74, 222, 128)',
                    }
                  : {
                      backgroundColor: `${segmentColor}10`,
                      borderColor: `${segmentColor}40`,
                      borderWidth: '1px',
                    }
              }
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  {/* Active indicator (only for active segment) */}
                  {isActive && (
                    <div className="flex items-center gap-2 mb-2">
                      <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                      <span className="text-xs text-green-400 font-medium uppercase tracking-wide">Active</span>
                    </div>
                  )}

                  <h3
                    className={`font-semibold ${isActive ? 'text-base' : 'text-sm'} mb-1 break-words`}
                    style={isActive ? {} : { color: segmentColor }}
                  >
                    {segment.topic || 'Unknown Topic'}
                  </h3>
                  <div className="flex items-center gap-3 text-xs text-[var(--text-tertiary)]">
                    <div className="flex items-center gap-1">
                      <Clock size={12} />
                      <span>{duration}</span>
                    </div>
                    <div className="flex items-center gap-1 cursor-help">
                      <MessageSquare size={12} />
                      <span>{isActive ? (currentSegment?.segment_message_count || 0) : segment.transcripts.length}</span>
                    </div>
                  </div>
                </div>
                <div
                  className="w-1 h-12 rounded-full"
                  style={{ backgroundColor: isActive ? 'rgb(74, 222, 128)' : segmentColor }}
                />
              </div>

              {/* Tooltip on hover */}
              <AnimatePresence>
                {hoveredSegment === segment.id && segment.transcripts.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    transition={{ duration: 0.2 }}
                    className="absolute left-0 right-0 top-full mt-2 z-50 glass-dark rounded-xl p-4 shadow-2xl border border-[var(--border-primary)]"
                    style={{ borderLeftColor: segmentColor, borderLeftWidth: '3px' }}
                  >
                    <h4 className="text-sm font-semibold mb-3" style={{ color: segmentColor }}>
                      Messages in this segment:
                    </h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {segment.transcripts.map((t, idx) => (
                        <div key={idx} className="text-xs min-w-0">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[var(--text-secondary)] font-medium truncate">
                              {t.speaker}
                            </span>
                            <span className="text-[var(--text-tertiary)] flex-shrink-0">{t.timestamp}</span>
                          </div>
                          <p className="text-[var(--text-primary)] leading-relaxed break-words">
                            {t.text}
                          </p>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          );
        })}
      </div>

      {allSegments.length > 0 && (
        <div className="mt-6 pt-4 border-t border-[var(--border-primary)]">
          <div className="text-xs text-[var(--text-tertiary)] text-center">
            <span className="font-semibold text-[var(--text-secondary)]">
              {allSegments.length}
            </span>{' '}
            {allSegments.length === 1 ? 'segment' : 'segments'} total
          </div>
        </div>
      )}
    </div>
  );
}
