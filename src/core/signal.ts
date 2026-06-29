export type Signal = -1 | 0 | 1;

export function toSignal(value: number, threshold = 0): Signal {
  if (value > threshold) {
    return 1;
  }

  if (value < -threshold) {
    return -1;
  }

  return 0;
}

export function isActiveSignal(signal: Signal): boolean {
  return signal !== 0;
}

export function signalMagnitude(signal: Signal): number {
  return Math.abs(signal);
}

export function clampMagnitude(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function ema(previous: number, sample: number, alpha: number): number {
  return previous * (1 - alpha) + sample * alpha;
}
