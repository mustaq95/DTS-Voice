'use client';

import { motion } from 'framer-motion';
import { Lightbulb, AlertTriangle, CheckSquare } from 'lucide-react';
import { Nudge } from '../lib/types';
import { fadeInUp } from '../lib/animations';

interface NudgesPanelProps {
  nudges: Nudge[];
}

function NudgeCard({ nudge }: { nudge: Nudge }) {
  const percentage = Math.round(nudge.confidence * 100);

  const getTypeInfo = (type: string) => {
    switch (type) {
      case 'key_proposal':
        return {
          title: 'Key Proposal',
          icon: Lightbulb,
          color: 'var(--segment-1)', // Purple
          bgColor: 'rgba(102, 126, 234, 0.1)',
        };
      case 'delivery_risk':
        return {
          title: 'Delivery Risk',
          icon: AlertTriangle,
          color: 'var(--color-warning)', // Amber
          bgColor: 'rgba(245, 158, 11, 0.1)',
        };
      case 'action_item':
        return {
          title: 'Action Item',
          icon: CheckSquare,
          color: 'var(--color-success)', // Green
          bgColor: 'rgba(16, 185, 129, 0.1)',
        };
      default:
        return {
          title: type,
          icon: Lightbulb,
          color: 'var(--color-info)',
          bgColor: 'rgba(6, 182, 212, 0.1)',
        };
    }
  };

  const typeInfo = getTypeInfo(nudge.type);
  const Icon = typeInfo.icon;

  return (
    <motion.div
      {...fadeInUp}
      className="glass-dark rounded-xl p-4 border-l-4"
      style={{ borderLeftColor: typeInfo.color }}
      whileHover={{
        y: -2,
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.12)',
        transition: {
          duration: 0.2,
          ease: 'easeOut',
        },
      }}
    >
      <div className="flex items-start gap-3">
        <div
          className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ backgroundColor: typeInfo.bgColor }}
        >
          <Icon size={20} style={{ color: typeInfo.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between mb-2">
            <h3
              className="font-semibold text-sm"
              style={{ color: typeInfo.color }}
            >
              {typeInfo.title}
            </h3>
            <span className="text-xs text-[var(--text-tertiary)]">
              {percentage}% confidence
            </span>
          </div>
          <p className="text-[var(--text-primary)] text-sm mb-2">
            {nudge.title}
          </p>
          <p className="text-[var(--text-tertiary)] text-xs italic break-words">
            &quot;{nudge.quote}&quot;
          </p>
        </div>
      </div>
    </motion.div>
  );
}

export default function NudgesPanel({ nudges }: NudgesPanelProps) {
  return (
    <div className="glass rounded-2xl p-5 flex flex-col h-full">
      <h2 className="text-lg font-semibold mb-4">Nudges</h2>
      <div className="flex-1 overflow-y-auto space-y-3">
        {nudges.length === 0 ? (
          <p className="text-[var(--text-tertiary)] text-sm">No nudges yet...</p>
        ) : (
          nudges.map((nudge, index) => <NudgeCard key={index} nudge={nudge} />)
        )}
      </div>
    </div>
  );
}
