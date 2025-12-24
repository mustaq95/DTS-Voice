'use client';

import { Clock, MessageSquare, Zap } from 'lucide-react';
import { SegmentUpdate } from '../lib/types';

interface ClassificationStatusProps {
  segmentUpdate: SegmentUpdate;
}

export default function ClassificationStatus({ segmentUpdate }: ClassificationStatusProps) {
  const { current_topic, classification_status, segment_message_count, transcripts_in_buffer } =
    segmentUpdate;

  // Determine status color based on classification action
  const getStatusColor = (action?: string) => {
    if (!action) return 'text-[var(--text-tertiary)]';
    switch (action) {
      case 'NEW_TOPIC':
        return 'text-green-400';
      case 'CONTINUE':
        return 'text-blue-400';
      case 'NOISE':
        return 'text-gray-400';
      default:
        return 'text-yellow-400';
    }
  };

  // Get background color for action badge
  const getStatusBg = (action?: string) => {
    if (!action) return 'bg-gray-500/20';
    switch (action) {
      case 'NEW_TOPIC':
        return 'bg-green-500/20';
      case 'CONTINUE':
        return 'bg-blue-500/20';
      case 'NOISE':
        return 'bg-gray-500/20';
      default:
        return 'bg-yellow-500/20';
    }
  };

  return (
    <div className="glass rounded-2xl p-4 flex-shrink-0">
      {/* Current Segment Topic */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-1">
          <Zap size={14} className="text-[var(--color-accent)]" />
          <span className="text-xs text-[var(--text-tertiary)] uppercase tracking-wide">
            Current Segment
          </span>
        </div>
        <h3 className="text-lg font-semibold text-[var(--text-primary)] break-words mb-2">
          {current_topic || 'No active segment'}
        </h3>

        {/* Segment metrics */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1">
            <MessageSquare size={12} className="text-[var(--text-tertiary)]" />
            <span className="text-xs text-[var(--text-secondary)]">
              {segment_message_count || 0} message{segment_message_count !== 1 ? 's' : ''}
            </span>
          </div>

          {/* Pending transcripts badge */}
          {transcripts_in_buffer !== undefined && (
            <span className="px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 text-xs font-medium flex items-center gap-1">
              <Clock size={10} />
              {transcripts_in_buffer} pending
            </span>
          )}
        </div>
      </div>

      {/* Classification Status */}
      {classification_status && (
        <div className="border-t border-[var(--color-border)] pt-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-[var(--text-tertiary)] uppercase tracking-wide">
              Last Classification
            </span>
            <span
              className={`px-2 py-0.5 rounded-full ${getStatusBg(classification_status.action)} ${getStatusColor(classification_status.action)} text-xs font-medium`}
            >
              {classification_status.action}
            </span>
          </div>

          {/* Classified transcript */}
          {classification_status.classified_transcript && (
            <p className="text-sm text-[var(--text-primary)] break-words mb-2 p-2 rounded bg-[var(--color-surface)] bg-opacity-30">
              &ldquo;{classification_status.classified_transcript}&rdquo;
            </p>
          )}

          {/* Classification details */}
          <div className="flex flex-col gap-1.5 text-xs">
            {/* Topic */}
            <div className="flex items-start gap-2">
              <span className="text-[var(--text-tertiary)] flex-shrink-0 min-w-[50px]">
                Topic:
              </span>
              <span className="text-[var(--text-primary)] font-medium break-words">
                {classification_status.topic}
              </span>
            </div>

            {/* Reason */}
            <div className="flex items-start gap-2">
              <span className="text-[var(--text-tertiary)] flex-shrink-0 min-w-[50px]">
                Reason:
              </span>
              <span className="text-[var(--text-secondary)] italic break-words">
                {classification_status.reason}
              </span>
            </div>

            {/* Timestamp */}
            <div className="flex items-center gap-2">
              <span className="text-[var(--text-tertiary)] flex-shrink-0 min-w-[50px]">
                Time:
              </span>
              <span className="text-[var(--text-secondary)]">
                {new Date(classification_status.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
