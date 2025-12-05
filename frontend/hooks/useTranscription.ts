import { useState, useEffect, useCallback } from 'react';
import { Transcript, Nudge } from '../lib/types';

export function useTranscription() {
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [nudges, setNudges] = useState<Nudge[]>([]);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws/transcripts';
    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    websocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === 'transcript') {
          setTranscripts(prev => [...prev, message.data]);
        } else if (message.type === 'nudge') {
          setNudges(prev => [...prev, message.data]);
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    websocket.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      websocket.close();
    };
  }, []);

  const requestNudges = useCallback(async () => {
    const recentTranscripts = transcripts
      .filter(t => t.is_final)
      .slice(-5)
      .map(t => t.text);

    if (recentTranscripts.length === 0) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
      const response = await fetch(`${apiUrl}/nudge`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcripts: recentTranscripts
        })
      });

      const data = await response.json();
      if (data.nudges) {
        setNudges(prev => [...prev, ...data.nudges]);
      }
    } catch (error) {
      console.error('Error requesting nudges:', error);
    }
  }, [transcripts]);

  return {
    transcripts,
    nudges,
    connected,
    requestNudges
  };
}
