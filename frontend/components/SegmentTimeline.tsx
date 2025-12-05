'use client';

import { useState } from 'react';
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

  return (
    <div className="glass rounded-2xl p-5">
      <div className="flex items-center gap-2 mb-5">
        <Clock size={18} className="text-[var(--color-active)]" />
        <h2 className="text-lg font-semibold">Segment Timeline</h2>
      </div>

      {segments.length === 0 && !currentSegment && (
        <p className="text-[var(--text-tertiary)] text-sm text-center py-8">
          No segments yet. Start speaking to see topics appear.
        </p>
      )}

      <div className="space-y-3">
        {segments.map((segment, index) => {
          const segmentColor = SEGMENT_COLORS[index % SEGMENT_COLORS.length];
          const duration = formatDuration(segment.started_at, segment.completed_at);

          return (
            <motion.div
              key={segment.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              onClick={() => onSegmentClick?.(segment.id)}
              onMouseEnter={() => setHoveredSegment(segment.id)}
              onMouseLeave={() => setHoveredSegment(null)}
              className={`relative p-4 rounded-xl border transition-all ${
                onSegmentClick
                  ? 'cursor-pointer hover:bg-[var(--color-surface)] hover:border-opacity-30 hover:scale-[1.02]'
                  : ''
              }`}
              style={{
                backgroundColor: `${segmentColor}10`,
                borderColor: `${segmentColor}40`,
                borderWidth: '1px',
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                  <h3
                    className="font-semibold text-sm mb-1 truncate"
                    style={{ color: segmentColor }}
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
                      <span>{segment.transcripts.length}</span>
                    </div>
                  </div>
                </div>
                <div
                  className="w-1 h-12 rounded-full"
                  style={{ backgroundColor: segmentColor }}
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
                    className="absolute left-0 top-full mt-2 w-80 z-50 glass-dark rounded-xl p-4 shadow-2xl border border-[var(--border-primary)]"
                    style={{ borderLeftColor: segmentColor, borderLeftWidth: '3px' }}
                  >
                    <h4 className="text-sm font-semibold mb-3" style={{ color: segmentColor }}>
                      Messages in this segment:
                    </h4>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                      {segment.transcripts.map((t, idx) => (
                        <div key={idx} className="text-xs">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="text-[var(--text-secondary)] font-medium">
                              {t.speaker}
                            </span>
                            <span className="text-[var(--text-tertiary)]">{t.timestamp}</span>
                          </div>
                          <p className="text-[var(--text-primary)] leading-relaxed">
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

        {/* Current active segment (from buffer_update) */}
        {currentSegment && currentSegment.classification_status && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 rounded-xl border shadow-lg"
            style={{
              backgroundColor: 'rgba(59, 130, 246, 0.15)',
              borderColor: 'var(--color-active)',
              borderWidth: '2px',
            }}
          >
            <div className="flex-1 min-w-0">
              {/* Header with topic and action badge */}
              <div className="flex items-center justify-between gap-2 mb-3">
                <h3 className="font-semibold text-base text-[var(--color-active)] flex-1 min-w-0 truncate">
                  {currentSegment.current_topic}
                </h3>
                <span
                  className="text-xs font-semibold px-2.5 py-1 rounded flex-shrink-0"
                  style={getActionColor(currentSegment.classification_status.action)}
                >
                  {currentSegment.classification_status.action}
                </span>
              </div>

              {/* Classified transcript */}
              {currentSegment.classification_status.classified_transcript && (
                <p className="text-sm text-[var(--text-primary)] mb-2 leading-relaxed truncate">
                  "{currentSegment.classification_status.classified_transcript}"
                </p>
              )}

              {/* Classification reason */}
              <p className="text-xs text-[var(--text-tertiary)] italic mb-3 leading-relaxed">
                {currentSegment.classification_status.reason}
              </p>

              {/* Real-time metrics */}
              <div className="flex items-center gap-3 text-xs text-[var(--text-secondary)]">
                <div className="flex items-center gap-1">
                  <Clock size={12} />
                  <span>{liveDuration}</span>
                </div>
                <div className="flex items-center gap-1">
                  <MessageSquare size={12} />
                  <span>
                    {currentSegment.segment_message_count || 0} message
                    {currentSegment.segment_message_count !== 1 ? 's' : ''}
                  </span>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {segments.length > 0 && (
        <div className="mt-6 pt-4 border-t border-[var(--border-primary)]">
          <div className="text-xs text-[var(--text-tertiary)] text-center">
            <span className="font-semibold text-[var(--text-secondary)]">
              {segments.length + (currentSegment ? 1 : 0)}
            </span>{' '}
            {(segments.length + (currentSegment ? 1 : 0)) === 1 ? 'segment' : 'segments'} total
          </div>
        </div>
      )}
    </div>
  );
}
