/**
 * Framer Motion animation variants for the application
 * Provides consistent, reusable animations across components
 */

export const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
  exit: { opacity: 0, y: -20, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
};

export const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit: { opacity: 0 },
  transition: { duration: 0.2 },
};

export const slideInRight = {
  initial: { opacity: 0, x: 20 },
  animate: { opacity: 1, x: 0, transition: { type: 'spring' as const, stiffness: 400, damping: 25 } },
  exit: { opacity: 0, x: -20, transition: { type: 'spring' as const, stiffness: 400, damping: 25 } },
};

export const slideInLeft = {
  initial: { opacity: 0, x: -20 },
  animate: { opacity: 1, x: 0, transition: { type: 'spring' as const, stiffness: 400, damping: 25 } },
  exit: { opacity: 0, x: 20, transition: { type: 'spring' as const, stiffness: 400, damping: 25 } },
};

export const scaleIn = {
  initial: { scale: 0.95, opacity: 0 },
  animate: { scale: 1, opacity: 1, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
  exit: { scale: 0.95, opacity: 0, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
};

export const staggerContainer = {
  initial: {},
  animate: {
    transition: {
      staggerChildren: 0.05,
      delayChildren: 0.1,
    },
  },
};

export const staggerItem = {
  initial: { opacity: 0, y: 10 },
  animate: { opacity: 1, y: 0, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
};

export const pulseAnimation = {
  scale: [1, 1.05, 1],
  opacity: [1, 0.8, 1],
  transition: {
    duration: 2,
    repeat: Infinity,
  },
};

export const shimmerAnimation = {
  backgroundPosition: ['200% 0', '-200% 0'],
  transition: {
    duration: 2,
    repeat: Infinity,
  },
};

export const gradientAnimation = {
  backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
  transition: {
    duration: 3,
    repeat: Infinity,
  },
};

export const waveAnimation = {
  y: [0, -10, 0],
  transition: {
    duration: 1.5,
    repeat: Infinity,
  },
};

export const bounceAnimation = {
  y: [0, -5, 0],
  transition: {
    duration: 0.6,
    repeat: Infinity,
  },
};

// Button press animation
export const buttonTap = {
  scale: 0.95,
  transition: { duration: 0.1 },
};

// Card hover animation
export const cardHover = {
  y: -4,
  boxShadow: '0 10px 40px rgba(0, 0, 0, 0.3)',
  transition: { duration: 0.2 },
};

// Page transition variants
export const pageTransition = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.3 } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.3 } },
};

// List item variants
export const listItem = {
  hidden: { opacity: 0, x: -10 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { type: 'spring' as const, stiffness: 500, damping: 30 },
  },
  exit: { opacity: 0, x: 10, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
};

// Modal variants
export const modalBackdrop = {
  hidden: { opacity: 0 },
  visible: { opacity: 1 },
  exit: { opacity: 0 },
};

export const modalContent = {
  hidden: { scale: 0.95, opacity: 0, y: 20 },
  visible: {
    scale: 1,
    opacity: 1,
    y: 0,
    transition: { type: 'spring' as const, stiffness: 500, damping: 30 },
  },
  exit: { scale: 0.95, opacity: 0, y: 20, transition: { type: 'spring' as const, stiffness: 500, damping: 30 } },
};

// Skeleton loading shimmer
export const skeletonShimmer = {
  backgroundPosition: ['-200%', '200%'],
  transition: {
    duration: 1.5,
    repeat: Infinity,
  },
};

// Live indicator pulse
export const livePulse = {
  scale: [1, 1.2, 1],
  opacity: [1, 0.6, 1],
  transition: {
    duration: 1.5,
    repeat: Infinity,
  },
};
