export interface Transcript {
  timestamp: string;
  speaker: string;
  text: string;
  is_final: boolean;
}

export interface Nudge {
  type: 'key_proposal' | 'delivery_risk' | 'action_item';
  title: string;
  quote: string;
  confidence: number;
  segment_id?: string;  // Optional: Track which segment generated this nudge
}

export interface Segment {
  id: string;
  topic: string;
  transcripts: Transcript[];
  status: 'active' | 'completed';
  started_at: string;
  completed_at?: string;
}

export interface ClassificationStatus {
  action: 'CONTINUE' | 'NEW_TOPIC' | 'NOISE';
  topic: string;
  reason: string;
  timestamp: string;
  classified_transcript?: string;
}

export interface NoiseItem {
  transcript: Transcript;
  reason: string;
  filtered_at: string;
}

export interface SegmentUpdate {
  current_topic: string;
  classification_status?: ClassificationStatus;
  segment_started_at?: string;
  segment_message_count?: number;
  // Legacy fields (for backward compatibility)
  transcripts_in_buffer?: number;
  total_segments?: number;
  waiting_transcripts?: Transcript[];
}

export interface MeetingSession {
  meeting_id: string;
  room_name: string;
  started_at: string;
  transcripts: Transcript[];
  nudges: Nudge[];
  segments?: Segment[];
}

export interface WebSocketMessage {
  type: 'transcript' | 'nudge' | 'status' | 'segment_complete' | 'buffer_update' | 'noise_filtered';
  data: Transcript | Nudge | Segment | SegmentUpdate | NoiseItem | unknown;
}
