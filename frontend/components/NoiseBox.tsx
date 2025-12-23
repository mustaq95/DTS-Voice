'use client';

import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, Volume2 } from 'lucide-react';
import { NoiseItem } from '../lib/types';

interface NoiseBoxProps {
  noiseItems: NoiseItem[];
}

function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
}

export default function NoiseBox({ noiseItems }: NoiseBoxProps) {
  return (
    <div className="glass rounded-2xl p-5 max-h-[200px] flex-shrink-0 flex flex-col">
      <div className="flex items-center justify-between mb-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          <AlertCircle size={18} className="text-[var(--color-warning)]" />
          <h2 className="text-lg font-semibold">Filtered Noise</h2>
          <span className="px-2 py-0.5 rounded-full bg-[var(--color-warning)] bg-opacity-20 text-[var(--color-warning)] text-xs font-medium">
            {noiseItems.length}
          </span>
        </div>
      </div>

      {noiseItems.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-center flex-1">
          <div className="w-12 h-12 rounded-full bg-[var(--color-surface)] flex items-center justify-center mb-3">
            <Volume2 size={24} className="text-[var(--text-tertiary)]" />
          </div>
          <p className="text-sm text-[var(--text-tertiary)]">
            No noise filtered yet
          </p>
        </div>
      ) : (
        <div className="space-y-2 overflow-y-auto flex-1">
          <AnimatePresence mode="popLayout">
            {noiseItems
              .slice()
              .reverse()
              .map((item, index) => (
                <motion.div
                  key={`${item.filtered_at}-${index}`}
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="p-3 rounded-lg border border-[var(--color-warning)] border-opacity-30 bg-[var(--color-warning)] bg-opacity-5 hover:bg-opacity-10 transition-all"
                >
                  <div className="flex items-start justify-between gap-2 mb-1">
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="text-xs font-medium text-[var(--text-secondary)]">
                        {item.transcript.speaker || 'Unknown'}
                      </span>
                      <span className="text-xs text-[var(--text-tertiary)]">
                        {item.transcript.timestamp}
                      </span>
                    </div>
                    <span className="text-xs text-[var(--color-warning)] flex-shrink-0">
                      {formatTimestamp(item.filtered_at)}
                    </span>
                  </div>
                  <p className="text-xs text-[var(--text-primary)] mb-1 line-clamp-2 leading-relaxed">
                    "{item.transcript.text}"
                  </p>
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-[var(--text-tertiary)]">Reason:</span>
                    <span className="text-xs text-[var(--color-warning)] italic">
                      {item.reason}
                    </span>
                  </div>
                </motion.div>
              ))}
          </AnimatePresence>
        </div>
      )}
    </div>
  );
}
