'use client';

import { motion } from 'framer-motion';
import { livePulse } from '../lib/animations';

interface LiveIndicatorProps {
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

export default function LiveIndicator({
  label = 'LIVE',
  size = 'md',
  className = ''
}: LiveIndicatorProps) {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  };

  const textSizeClasses = {
    sm: 'text-xs',
    md: 'text-sm',
    lg: 'text-base',
  };

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <motion.div
        className={`${sizeClasses[size]} rounded-full bg-[var(--color-live)]`}
        animate={livePulse}
      />
      <span className={`${textSizeClasses[size]} font-semibold text-[var(--color-live)] uppercase tracking-wide`}>
        {label}
      </span>
    </div>
  );
}
