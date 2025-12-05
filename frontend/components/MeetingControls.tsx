'use client';

import { motion } from 'framer-motion';
import { Mic, MicOff, Download, PhoneOff } from 'lucide-react';
import LiveIndicator from './LiveIndicator';
import { buttonTap } from '../lib/animations';

interface MeetingControlsProps {
  micOn: boolean;
  isConnected: boolean;
  onMicToggle: () => void;
  onExport: () => void;
  onEndMeeting: () => void;
  transcriptCount?: number;
  segmentCount?: number;
}

export default function MeetingControls({
  micOn,
  isConnected,
  onMicToggle,
  onExport,
  onEndMeeting,
  transcriptCount = 0,
  segmentCount = 0,
}: MeetingControlsProps) {
  return (
    <div className="glass-dark px-6 py-4">
      <div className="flex justify-between items-center">
        {/* Left: Status and Stats */}
        <div className="flex items-center gap-6">
          {isConnected && micOn && <LiveIndicator size="sm" label="RECORDING" />}
          {isConnected && !micOn && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[var(--color-warning)]" />
              <span className="text-sm font-semibold text-[var(--color-warning)] uppercase tracking-wide">
                PAUSED
              </span>
            </div>
          )}
          {!isConnected && (
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-[var(--text-tertiary)]" />
              <span className="text-sm font-semibold text-[var(--text-tertiary)] uppercase tracking-wide">
                DISCONNECTED
              </span>
            </div>
          )}

          <div className="flex items-center gap-4 text-sm text-[var(--text-tertiary)]">
            <div>
              <span className="font-semibold text-[var(--text-secondary)]">
                {transcriptCount}
              </span>{' '}
              messages
            </div>
            <div className="w-1 h-1 rounded-full bg-[var(--text-tertiary)]" />
            <div>
              <span className="font-semibold text-[var(--text-secondary)]">
                {segmentCount}
              </span>{' '}
              segments
            </div>
          </div>
        </div>

        {/* Right: Action Buttons */}
        <div className="flex gap-3">
          {/* Microphone Toggle */}
          <motion.button
            whileTap={buttonTap}
            onClick={onMicToggle}
            disabled={!isConnected}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
              micOn
                ? 'bg-[var(--color-live)] hover:bg-red-600 text-white shadow-lg'
                : 'glass hover:bg-[var(--color-surface)] text-[var(--text-primary)]'
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {micOn ? <Mic size={18} /> : <MicOff size={18} />}
            <span className="text-sm">{micOn ? 'Mic On' : 'Mic Off'}</span>
          </motion.button>

          {/* Export Button */}
          <motion.button
            whileTap={buttonTap}
            onClick={onExport}
            disabled={transcriptCount === 0}
            className="glass hover:bg-[var(--color-surface)] text-[var(--text-primary)] flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Download size={18} />
            <span className="text-sm">Export</span>
          </motion.button>

          {/* End Meeting Button */}
          <motion.button
            whileTap={buttonTap}
            onClick={onEndMeeting}
            disabled={!isConnected}
            className="bg-[var(--color-live)] hover:bg-red-600 text-white flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg"
          >
            <PhoneOff size={18} />
            <span className="text-sm">End</span>
          </motion.button>
        </div>
      </div>
    </div>
  );
}
