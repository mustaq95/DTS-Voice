'use client';

import { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, MessageSquare } from 'lucide-react';
import { Transcript, Segment } from '../lib/types';
import TranscriptMessage from './TranscriptMessage';
import SegmentHeader from './SegmentHeader';
import { staggerContainer } from '../lib/animations';

interface LiveTranscriptProps {
  transcripts: Transcript[];
  segments: Segment[];
}

export default function LiveTranscript({ transcripts, segments }: LiveTranscriptProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new transcripts arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [transcripts, autoScroll]);

  // Detect manual scroll and disable auto-scroll
  const handleScroll = () => {
    if (!scrollContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = scrollContainerRef.current;
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;

    setAutoScroll(isNearBottom);
  };

  // Filter transcripts by search query
  const filteredTranscripts = searchQuery
    ? transcripts.filter(
        (t) =>
          t.text.toLowerCase().includes(searchQuery.toLowerCase()) ||
          t.speaker.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : transcripts;

  // Group transcripts by segment
  const groupedTranscripts: Array<{
    segment: Segment | null;
    transcripts: Transcript[];
  }> = [];

  if (segments.length === 0) {
    // No segments yet - show all transcripts ungrouped
    groupedTranscripts.push({ segment: null, transcripts: filteredTranscripts });
  } else {
    // Build a set of transcript identifiers that are in segments
    // Use both timestamp AND text to uniquely identify transcripts
    const groupedIdentifiers = new Set(
      segments.flatMap((s) =>
        s.transcripts.map((t) => `${t.timestamp}|${t.text}`)
      )
    );

    // Group transcripts by their segments
    segments.forEach((segment) => {
      const segmentTranscripts = filteredTranscripts.filter((t) =>
        segment.transcripts.some((st) => st.timestamp === t.timestamp && st.text === t.text)
      );

      if (segmentTranscripts.length > 0) {
        groupedTranscripts.push({ segment, transcripts: segmentTranscripts });
      }
    });

    // Add ungrouped transcripts (not in any completed segment)
    // This includes: noise-filtered transcripts, transcripts waiting for classification, etc.
    const ungroupedTranscripts = filteredTranscripts.filter(
      (t) => !groupedIdentifiers.has(`${t.timestamp}|${t.text}`)
    );

    if (ungroupedTranscripts.length > 0) {
      groupedTranscripts.push({ segment: null, transcripts: ungroupedTranscripts });
    }
  }

  return (
    <div className="relative flex-1 glass rounded-2xl flex flex-col h-full">
      {/* Header */}
      <div className="p-5 pb-4 border-b border-[var(--border-primary)] flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <MessageSquare size={20} className="text-[var(--color-active)]" />
            <h2 className="text-xl font-semibold">Live Transcript</h2>
          </div>
          <div className="text-sm text-[var(--text-tertiary)]">
            {transcripts.length} {transcripts.length === 1 ? 'message' : 'messages'}
          </div>
        </div>

        {/* Search bar */}
        <div className="relative">
          <Search
            size={16}
            className="absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-tertiary)]"
          />
          <input
            type="text"
            placeholder="Search transcripts..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-transparent border border-[var(--border-primary)] rounded-lg pl-10 pr-4 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-tertiary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-active)] focus:ring-opacity-50 transition-all"
          />
        </div>
      </div>

      {/* Transcript list */}
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-5 space-y-4 min-w-0"
      >
        {filteredTranscripts.length === 0 && !searchQuery && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <div className="w-16 h-16 rounded-full bg-[var(--color-surface)] flex items-center justify-center mb-4">
              <MessageSquare size={32} className="text-[var(--text-tertiary)]" />
            </div>
            <p className="text-[var(--text-secondary)] mb-2">No transcripts yet</p>
            <p className="text-sm text-[var(--text-tertiary)] max-w-sm">
              Connect to the room and enable your microphone to start transcribing
            </p>
          </div>
        )}

        {filteredTranscripts.length === 0 && searchQuery && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <p className="text-[var(--text-secondary)] mb-2">No results found</p>
            <p className="text-sm text-[var(--text-tertiary)]">
              Try a different search term
            </p>
          </div>
        )}

        <AnimatePresence mode="popLayout">
          {groupedTranscripts.map((group, groupIndex) => (
            <motion.div
              key={group.segment?.id || `ungrouped-${groupIndex}`}
              variants={staggerContainer}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {group.segment && (
                <SegmentHeader
                  topic={group.segment.topic}
                  timestamp={group.segment.started_at}
                  isActive={group.segment.status === 'active'}
                  transcriptCount={group.transcripts.length}
                  colorIndex={groupIndex}
                />
              )}
              <div className="space-y-2">
                {group.transcripts.map((transcript, index) => (
                  <TranscriptMessage
                    key={`${transcript.timestamp}-${index}`}
                    transcript={transcript}
                  />
                ))}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        <div ref={bottomRef} />
      </div>

      {/* Scroll to bottom indicator */}
      {!autoScroll && (
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          onClick={() => {
            setAutoScroll(true);
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
          }}
          className="absolute bottom-6 left-1/2 -translate-x-1/2 glass-dark px-4 py-2 rounded-full text-sm font-medium text-[var(--text-primary)] hover:bg-[var(--color-surface)] transition-all shadow-lg"
        >
          ↓ New messages
        </motion.button>
      )}
    </div>
  );
}
