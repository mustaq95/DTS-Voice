'use client';

import { useEffect, useState, useCallback } from 'react';
import {
  Room,
  RoomEvent,
  LocalParticipant,
  createLocalAudioTrack,
  DataPacket_Kind,
  RemoteParticipant,
} from 'livekit-client';
import { Transcript, Segment, SegmentUpdate } from '../lib/types';

interface UseLiveKitProps {
  roomName: string;
  participantName: string;
}

interface UseLiveKitReturn {
  room: Room | null;
  isConnecting: boolean;
  isConnected: boolean;
  error: string | null;
  localParticipant: LocalParticipant | null;
  transcripts: Transcript[];
  segments: Segment[];
  segmentUpdate: SegmentUpdate | null;
  connectToRoom: () => Promise<void>;
  disconnectFromRoom: () => Promise<void>;
  enableMicrophone: () => Promise<void>;
  disableMicrophone: () => Promise<void>;
  isMicEnabled: boolean;
}

export function useLiveKit({
  roomName,
  participantName,
}: UseLiveKitProps): UseLiveKitReturn {
  const [room, setRoom] = useState<Room | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [localParticipant, setLocalParticipant] = useState<LocalParticipant | null>(null);
  const [isMicEnabled, setIsMicEnabled] = useState(false);
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [segmentUpdate, setSegmentUpdate] = useState<SegmentUpdate | null>(null);

  const connectToRoom = useCallback(async () => {
    if (room && room.state === 'connected') {
      console.log('Already connected to room');
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      // Get access token from backend
      const response = await fetch('http://localhost:8000/api/livekit/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          room_name: roomName,
          participant_name: participantName,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to get access token: ${response.statusText}`);
      }

      const { token, url } = await response.json();

      // Create and connect to room
      const newRoom = new Room();

      // Set up event listeners
      newRoom.on(RoomEvent.Connected, () => {
        console.log('Connected to room:', roomName);
        setIsConnected(true);
        setLocalParticipant(newRoom.localParticipant);
      });

      newRoom.on(RoomEvent.Disconnected, () => {
        console.log('Disconnected from room');
        setIsConnected(false);
        setIsMicEnabled(false);
      });

      newRoom.on(RoomEvent.Reconnecting, () => {
        console.log('Reconnecting to room...');
      });

      newRoom.on(RoomEvent.Reconnected, () => {
        console.log('Reconnected to room');
      });

      // Listen for data received (transcripts and segments from agent)
      newRoom.on(RoomEvent.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant, kind?: DataPacket_Kind, topic?: string) => {
        try {
          const decoder = new TextDecoder();
          const message = JSON.parse(decoder.decode(payload));

          console.log('🔔 Data received from agent:', message, 'topic:', topic);

          if (message.type === 'transcript' && message.data) {
            console.log('✅ Adding transcript:', message.data);
            setTranscripts((prev) => [...prev, message.data]);
          } else if (message.type === 'buffer_update') {
            console.log('📊 Segment buffer updated:', message.data);
            setSegmentUpdate(message.data);
          } else if (message.type === 'segment_complete') {
            console.log('✅ Segment completed:', message.data);
            setSegments((prev) => {
              // Check if segment already exists (prevent duplicates)
              const exists = prev.some((s) => s.id === message.data.id);
              if (exists) {
                return prev;
              }
              return [...prev, { ...message.data, status: 'completed' }];
            });
          } else if (message.type === 'noise_filtered') {
            console.log('🔇 Noise filtered:', message.data);
            // Optionally show filtered noise in UI for debugging
          } else {
            console.log('❌ Message type not recognized or no data:', message);
          }
        } catch (error) {
          console.error('Error parsing data packet:', error);
        }
      });

      // Connect to the room
      await newRoom.connect(url, token);
      setRoom(newRoom);
      setIsConnecting(false);
    } catch (err) {
      console.error('Error connecting to room:', err);
      setError(err instanceof Error ? err.message : 'Failed to connect');
      setIsConnecting(false);
    }
  }, [room, roomName, participantName]);

  const disconnectFromRoom = useCallback(async () => {
    if (room) {
      await room.disconnect();
      setRoom(null);
      setIsConnected(false);
      setLocalParticipant(null);
      setIsMicEnabled(false);
      setTranscripts([]);
      setSegments([]);
      setSegmentUpdate(null);
    }
  }, [room]);

  const enableMicrophone = useCallback(async () => {
    if (!room || !room.localParticipant) {
      console.error('Cannot enable microphone: not connected to room');
      return;
    }

    try {
      console.log('Requesting microphone access...');

      // First, check if we can get media devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioDevices = devices.filter(d => d.kind === 'audioinput');
      console.log('Available audio input devices:', audioDevices);

      if (audioDevices.length === 0) {
        throw new Error('No microphone devices found on this system');
      }

      // Request microphone permission and create audio track
      // Start with most basic approach
      let audioTrack;
      try {
        console.log('Trying to create audio track with default settings...');
        audioTrack = await createLocalAudioTrack();
      } catch (initialErr) {
        console.error('Basic audio track failed:', initialErr);
        throw initialErr;
      }

      console.log('Audio track created:', audioTrack);

      // Publish the audio track to the room
      await room.localParticipant.publishTrack(audioTrack);
      setIsMicEnabled(true);
      console.log('Microphone enabled and track published');
    } catch (err) {
      console.error('Error enabling microphone:', err);

      // More helpful error message
      if (err instanceof Error) {
        if (err.name === 'NotFoundError') {
          setError('No microphone found. Please check your microphone is connected and browser has permission.');
        } else if (err.name === 'NotAllowedError') {
          setError('Microphone permission denied. Please allow microphone access in your browser.');
        } else {
          setError(err.message);
        }
      } else {
        setError('Failed to enable microphone');
      }
    }
  }, [room]);

  const disableMicrophone = useCallback(async () => {
    if (!room || !room.localParticipant) {
      return;
    }

    // Unpublish all audio tracks
    const audioTracks = Array.from(room.localParticipant.audioTrackPublications.values());
    for (const publication of audioTracks) {
      if (publication.track) {
        publication.track.stop();
        await room.localParticipant.unpublishTrack(publication.track);
      }
    }

    setIsMicEnabled(false);
    console.log('Microphone disabled');
  }, [room]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (room) {
        room.disconnect();
      }
    };
  }, [room]);

  return {
    room,
    isConnecting,
    isConnected,
    error,
    localParticipant,
    transcripts,
    segments,
    segmentUpdate,
    connectToRoom,
    disconnectFromRoom,
    enableMicrophone,
    disableMicrophone,
    isMicEnabled,
  };
}
