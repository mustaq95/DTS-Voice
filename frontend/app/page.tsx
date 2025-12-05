'use client';

import { useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Toaster, toast } from 'sonner';
import { RefreshCw } from 'lucide-react';
import LiveTranscript from '../components/LiveTranscript';
import SegmentTimeline from '../components/SegmentTimeline';
import SegmentAnalytics from '../components/SegmentAnalytics';
import NudgesPanel from '../components/NudgesPanel';
import MeetingControls from '../components/MeetingControls';
import LiveIndicator from '../components/LiveIndicator';
import { useLiveKit } from '../hooks/useLiveKit';
import { Nudge } from '../lib/types';
import { pageTransition, buttonTap } from '../lib/animations';

export default function Home() {
  const [currentTime, setCurrentTime] = useState('');
  const [participantName] = useState(`user-${Math.random().toString(36).substring(7)}`);
  const [nudges, setNudges] = useState<Nudge[]>([]);

  // LiveKit connection
  const {
    isConnected,
    isConnecting,
    error,
    transcripts,
    segments,
    segmentUpdate,
    connectToRoom,
    disconnectFromRoom,
    enableMicrophone,
    disableMicrophone,
    isMicEnabled,
  } = useLiveKit({
    roomName: 'voice-fest',
    participantName,
  });

  // Function to request nudges from API for a specific segment
  const requestNudgesForSegment = useCallback(async (segment: any) => {
    // Extract transcript text from segment
    const segmentTranscripts = segment.transcripts.map((t: any) => t.text);

    if (segmentTranscripts.length === 0) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
      console.log(`📊 Requesting nudges for segment "${segment.topic}" (${segmentTranscripts.length} transcripts)`);

      const response = await fetch(`${apiUrl}/nudge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcripts: segmentTranscripts,
          segment_id: segment.id,
          topic: segment.topic,
        }),
      });

      const data = await response.json();
      if (data.nudges) {
        console.log(`✅ Received ${data.nudges.length} nudges for segment "${segment.topic}"`);
        setNudges((prev) => [...prev, ...data.nudges]);
      }
    } catch (error) {
      console.error('Error requesting nudges:', error);
    }
  }, []);

  // Auto-request nudges when a segment is completed
  useEffect(() => {
    const completedSegments = segments.filter((s) => s.status === 'completed');
    const lastSegment = completedSegments[completedSegments.length - 1];

    if (lastSegment) {
      // Check if we've already processed this segment
      const segmentId = lastSegment.id;
      const alreadyProcessed = nudges.some((n: any) => n.segment_id === segmentId);

      if (!alreadyProcessed) {
        console.log(`🎯 New completed segment detected: "${lastSegment.topic}" (ID: ${segmentId})`);
        requestNudgesForSegment(lastSegment);
      }
    }
  }, [segments, nudges, requestNudgesForSegment]);

  // Auto-connect to LiveKit room on mount
  useEffect(() => {
    connectToRoom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  // Show error toast if there's an error
  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  // Show connection status toasts
  useEffect(() => {
    if (isConnected) {
      toast.success('Connected to room');
    }
  }, [isConnected]);

  // Update time display on client side
  useEffect(() => {
    const updateTime = () => {
      setCurrentTime(
        new Date().toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit',
          hour12: true,
        })
      );
    };
    updateTime();
    const interval = setInterval(updateTime, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleMicToggle = async () => {
    if (!isConnected) {
      toast.error('Not connected to room yet');
      return;
    }

    if (isMicEnabled) {
      await disableMicrophone();
      toast.info('Microphone disabled');
    } else {
      await enableMicrophone();
      toast.success('Microphone enabled');
    }
  };

  const handleExport = async () => {
    const data = {
      transcripts,
      segments,
      nudges,
      exported_at: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `meeting-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Meeting data exported');
  };

  const handleEndMeeting = async () => {
    if (confirm('Are you sure you want to end this meeting?')) {
      await disconnectFromRoom();
      toast.info('Meeting ended');
    }
  };

  const handleRedeploy = async () => {
    if (
      confirm(
        '🔄 Redeploy System\n\nThis will:\n• Stop all services (LiveKit, Backend, Agent, Frontend)\n• Restart everything from scratch\n• Clear current session\n\nPage will automatically reload in ~20 seconds.\n\nContinue?'
      )
    ) {
      try {
        toast.loading('Stopping all services...', { id: 'redeploy' });

        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

        // Set a longer timeout for the fetch request (15 seconds)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);

        try {
          const response = await fetch(`${apiUrl}/redeploy`, {
            method: 'POST',
            signal: controller.signal,
          });

          clearTimeout(timeoutId);

          if (!response.ok) {
            throw new Error('Redeployment failed');
          }

          toast.success('Services stopped. Restarting...', { id: 'redeploy' });
        } catch (fetchError) {
          // If fetch fails, it's likely because the backend is shutting down
          // This is expected behavior, so we continue with the reload
          console.log('Backend shutting down (expected):', fetchError);
          toast.success('Services restarting...', { id: 'redeploy' });
        }

        // Wait 20 seconds for services to fully restart before reloading
        toast.loading('Waiting for services to restart... (20s)', { id: 'redeploy' });

        let countdown = 20;
        const countdownInterval = setInterval(() => {
          countdown--;
          if (countdown > 0) {
            toast.loading(`Reloading in ${countdown} seconds...`, { id: 'redeploy' });
          }
        }, 1000);

        setTimeout(() => {
          clearInterval(countdownInterval);
          toast.success('Reloading page...', { id: 'redeploy' });
          setTimeout(() => {
            window.location.reload();
          }, 500);
        }, 20000);
      } catch (error) {
        console.error('Error during redeployment:', error);
        toast.error('Redeployment failed. Please try manually with ./start.sh', {
          id: 'redeploy',
        });
      }
    }
  };

  return (
    <motion.div
      {...pageTransition}
      className="h-screen flex flex-col bg-[var(--color-background)]"
    >
      <Toaster position="top-right" richColors />

      {/* Header */}
      <div className="glass-dark px-6 py-4 border-b border-[var(--border-primary)]">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold gradient-text">
              DTS - Voice
            </h1>
            <div className="flex items-center gap-3 mt-1">
              <p className="text-sm text-[var(--text-tertiary)]">
                Room: voice-fest
              </p>
              {isConnected && <LiveIndicator size="sm" label="LIVE" />}
            </div>
          </div>
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-[var(--color-success)]' : 'bg-[var(--color-live)]'
                }`}
              />
              <span className="text-[var(--text-secondary)]">
                {isConnecting ? 'Connecting...' : isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
            <span className="text-[var(--text-tertiary)]">{currentTime}</span>

            {/* Redeploy Button */}
            <motion.button
              whileTap={buttonTap}
              onClick={handleRedeploy}
              className="glass hover:bg-[var(--color-surface)] text-[var(--text-primary)] flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all border border-[var(--border-primary)] hover:border-[var(--color-active)]"
              title="Restart all services from scratch"
            >
              <RefreshCw size={16} />
              <span className="text-sm">Redeploy</span>
            </motion.button>
          </div>
        </div>
      </div>

      {/* Main Content - 3 Column Layout */}
      <div className="flex-1 flex gap-5 p-5 overflow-hidden min-h-0">
        {/* Left Column: Segment Timeline */}
        <div className="w-80 flex-shrink-0 overflow-y-auto">
          <SegmentTimeline
            segments={segments}
            currentSegment={segmentUpdate}
            transcripts={transcripts}
          />
        </div>

        {/* Center Column: Live Transcript (Main Focus) */}
        <div className="flex-1 min-w-0 flex">
          <LiveTranscript transcripts={transcripts} segments={segments} />
        </div>

        {/* Right Column: Analytics & Nudges */}
        <div className="w-96 flex-shrink-0 flex flex-col gap-6 overflow-y-auto">
          <SegmentAnalytics segments={segments} />
          <NudgesPanel nudges={nudges} />
        </div>
      </div>

      {/* Controls */}
      <MeetingControls
        micOn={isMicEnabled}
        isConnected={isConnected}
        onMicToggle={handleMicToggle}
        onExport={handleExport}
        onEndMeeting={handleEndMeeting}
        transcriptCount={transcripts.length}
        segmentCount={segments.length}
      />
    </motion.div>
  );
}
