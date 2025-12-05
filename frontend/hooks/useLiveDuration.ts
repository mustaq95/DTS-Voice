'use client';

import { useState, useEffect } from 'react';

/**
 * Custom hook to calculate and update duration in real-time
 * @param startedAt ISO timestamp string of when the segment started
 * @returns Formatted duration string (e.g., "2m 13s", "45s")
 */
export function useLiveDuration(startedAt: string | undefined): string {
  const [duration, setDuration] = useState('0s');

  useEffect(() => {
    if (!startedAt) {
      setDuration('0s');
      return;
    }

    const updateDuration = () => {
      const start = new Date(startedAt).getTime();
      const now = Date.now();
      const durationMs = now - start;

      const minutes = Math.floor(durationMs / 60000);
      const seconds = Math.floor((durationMs % 60000) / 1000);

      if (minutes > 0) {
        setDuration(`${minutes}m ${seconds}s`);
      } else {
        setDuration(`${seconds}s`);
      }
    };

    // Update immediately
    updateDuration();

    // Update every second
    const interval = setInterval(updateDuration, 1000);

    return () => clearInterval(interval);
  }, [startedAt]);

  return duration;
}
