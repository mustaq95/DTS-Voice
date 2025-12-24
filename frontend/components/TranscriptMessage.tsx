'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { User } from 'lucide-react';
import { Transcript } from '../lib/types';
import { fadeInUp } from '../lib/animations';

interface TranscriptMessageProps {
  transcript: Transcript;
  showAvatar?: boolean;
}

// Generate consistent color for speaker based on name
function getSpeakerColor(speaker: string): string {
  const colors = [
    '#667eea', // Purple
    '#4ade80', // Green
    '#f59e0b', // Amber
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#8b5cf6', // Violet
    '#14b8a6', // Teal
  ];

  let hash = 0;
  for (let i = 0; i < speaker.length; i++) {
    hash = speaker.charCodeAt(i) + ((hash << 5) - hash);
  }

  return colors[Math.abs(hash) % colors.length];
}

const TranscriptMessage = React.memo(function TranscriptMessage({
  transcript,
  showAvatar = true,
}: TranscriptMessageProps) {
  const speakerColor = getSpeakerColor(transcript.speaker);
  const isInterim = !transcript.is_final;

  return (
    <motion.div
      {...fadeInUp}
      className={`flex gap-3 p-4 rounded-xl transition-all ${
        isInterim
          ? 'bg-[var(--color-surface)] opacity-70'
          : 'bg-transparent hover:bg-[var(--color-surface)] hover:bg-opacity-50'
      }`}
    >
      {showAvatar && (
        <div
          className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: `${speakerColor}20` }}
        >
          <User size={16} style={{ color: speakerColor }} />
        </div>
      )}
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 mb-1 flex-wrap">
          <span
            className="font-semibold text-sm"
            style={{ color: speakerColor }}
          >
            {transcript.speaker}
          </span>
          <span className="text-xs text-[var(--text-tertiary)]">
            {transcript.timestamp}
          </span>
          {isInterim && (
            <motion.span
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity }}
              className="text-xs text-[var(--color-warning)]"
            >
              ⟳
            </motion.span>
          )}
        </div>
        <p className="text-[var(--text-primary)] leading-relaxed break-words">
          {transcript.text}
        </p>
      </div>
    </motion.div>
  );
});

export default TranscriptMessage;
