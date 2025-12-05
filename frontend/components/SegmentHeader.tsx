'use client';

import { motion } from 'framer-motion';
import { Hash } from 'lucide-react';
import LiveIndicator from './LiveIndicator';

interface SegmentHeaderProps {
  topic: string;
  timestamp: string;
  isActive?: boolean;
  transcriptCount?: number;
  colorIndex?: number;
}

const SEGMENT_COLORS = [
  'var(--segment-1)', // Purple
  'var(--segment-2)', // Green
  'var(--segment-3)', // Amber
  'var(--segment-4)', // Pink
  'var(--segment-5)', // Cyan
];

export default function SegmentHeader({
  topic,
  timestamp,
  isActive = false,
  transcriptCount = 0,
  colorIndex = 0,
}: SegmentHeaderProps) {
  const segmentColor = SEGMENT_COLORS[colorIndex % SEGMENT_COLORS.length];

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ type: 'spring', stiffness: 500, damping: 30 }}
      className="sticky top-0 z-10 mb-4"
    >
      <div className="glass-dark rounded-xl p-4 border-l-4 shadow-md" style={{ borderLeftColor: segmentColor }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: `${segmentColor}20` }}
            >
              <Hash size={20} style={{ color: segmentColor }} />
            </div>
            <div>
              <h3 className="text-lg font-semibold" style={{ color: segmentColor }}>
                {topic || 'Unknown Topic'}
              </h3>
              <div className="flex items-center gap-3 text-sm text-[var(--text-tertiary)] mt-1">
                <span>{new Date(timestamp).toLocaleTimeString()}</span>
                <span>•</span>
                <span>{transcriptCount} {transcriptCount === 1 ? 'message' : 'messages'}</span>
              </div>
            </div>
          </div>
          {isActive && <LiveIndicator size="sm" />}
        </div>
      </div>
    </motion.div>
  );
}
