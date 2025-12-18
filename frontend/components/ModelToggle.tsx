'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { Loader2 } from 'lucide-react';
import { buttonTap } from '../lib/animations';

interface ModelToggleProps {
  roomName: string;
  disabled?: boolean;
}

type TranscriptionEngine = 'mlx_whisper' | 'hamza';

export default function ModelToggle({ roomName, disabled = false }: ModelToggleProps) {
  const [currentModel, setCurrentModel] = useState<TranscriptionEngine>('mlx_whisper');
  const [isLoading, setIsLoading] = useState(false);

  // Fetch current model on mount
  useEffect(() => {
    const fetchCurrentModel = async () => {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
        const response = await fetch(`${apiUrl}/config/model/${roomName}`);
        const data = await response.json();
        setCurrentModel(data.current_model);
      } catch (error) {
        console.error('Failed to fetch current model:', error);
      }
    };
    fetchCurrentModel();
  }, [roomName]);

  const handleModelSwitch = async (model: TranscriptionEngine) => {
    if (model === currentModel || isLoading) return;

    setIsLoading(true);
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
      const response = await fetch(`${apiUrl}/config/model`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ room_name: roomName, model }),
      });

      const data = await response.json();
      if (data.status === 'success' || data.status === 'unchanged') {
        setCurrentModel(model);
        const modelName = model === 'mlx_whisper' ? 'MLX Whisper' : 'Hamza STT';
        toast.success(`Switched to ${modelName}`);
      } else {
        toast.error('Failed to switch model');
      }
    } catch (error) {
      console.error('Model switch error:', error);
      toast.error('Failed to switch model');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative flex gap-2 p-1 rounded-xl bg-[var(--color-surface)]/50">
      {/* Animated Background Slider */}
      <motion.div
        className="absolute inset-1 rounded-lg bg-[var(--color-active)] shadow-lg"
        initial={false}
        animate={{
          x: currentModel === 'mlx_whisper' ? 0 : 'calc(100% + 0.5rem)',
          width: 'calc(50% - 0.25rem)',
        }}
        transition={{
          type: 'spring',
          stiffness: 300,
          damping: 30,
        }}
      />

      {/* Whisper Button */}
      <motion.button
        whileTap={buttonTap}
        onClick={() => handleModelSwitch('mlx_whisper')}
        disabled={isLoading || disabled}
        className={`relative z-10 px-6 py-3 rounded-lg text-sm font-medium transition-colors ${
          currentModel === 'mlx_whisper'
            ? 'text-white'
            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
        } ${(isLoading || disabled) ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        {isLoading && currentModel !== 'mlx_whisper' ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          'Whisper'
        )}
      </motion.button>

      {/* Hamza Button */}
      <motion.button
        whileTap={buttonTap}
        onClick={() => handleModelSwitch('hamza')}
        disabled={isLoading || disabled}
        className={`relative z-10 px-6 py-3 rounded-lg text-sm font-medium transition-colors ${
          currentModel === 'hamza'
            ? 'text-white'
            : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
        } ${(isLoading || disabled) ? 'opacity-50 cursor-not-allowed' : ''}`}
      >
        {isLoading && currentModel !== 'hamza' ? (
          <Loader2 size={16} className="animate-spin" />
        ) : (
          'Hamza'
        )}
      </motion.button>
    </div>
  );
}
