'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, MessageSquare } from 'lucide-react';
import { Segment, SegmentUpdate, Transcript } from '../lib/types';
import LiveIndicator from './LiveIndicator';

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

export default function SegmentTimeline({
  segments,
  currentSegment,
  transcripts = [],
  onSegmentClick,
}: SegmentTimelineProps) {
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);

  // Get the latest transcript being classified
  const latestTranscript = transcripts.length > 0 ? transcripts[transcripts.length - 1] : null;

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
        {currentSegment && (
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
              <div className="flex items-center gap-2 mb-2">
                <h3 className="font-semibold text-base text-[var(--color-active)]">
                  {currentSegment.current_topic}
                </h3>
                <LiveIndicator size="sm" label="CLASSIFYING" />
              </div>
              <div className="flex items-center gap-3 text-xs text-[var(--text-secondary)] mb-3">
                <div className="flex items-center gap-1">
                  <MessageSquare size={12} />
                  <span>{currentSegment.transcripts_in_buffer} messages in buffer</span>
                </div>
              </div>
              {/* Show latest transcript being classified */}
              {latestTranscript && (
                <div className="mt-2 p-3 rounded-lg bg-[var(--color-surface)] bg-opacity-50 border border-[var(--border-primary)]">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-[var(--text-secondary)]">
                      {latestTranscript.speaker}
                    </span>
                    <span className="text-xs text-[var(--text-tertiary)]">
                      {latestTranscript.timestamp}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-primary)] leading-relaxed line-clamp-3">
                    &ldquo;{latestTranscript.text}&rdquo;
                  </p>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </div>

      {segments.length > 0 && (
        <div className="mt-6 pt-4 border-t border-[var(--border-primary)]">
          <div className="text-xs text-[var(--text-tertiary)] text-center">
            <span className="font-semibold text-[var(--text-secondary)]">
              {currentSegment?.total_segments || segments.length}
            </span>{' '}
            {(currentSegment?.total_segments || segments.length) === 1 ? 'segment' : 'segments'} total
          </div>
        </div>
      )}
    </div>
  );
}
