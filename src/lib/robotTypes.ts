export interface RobotType {
  id: string;
  name: string;
  description: string;
  sensorCount: number;
  motorCount: number;
  modelSize: 'small' | 'medium' | 'large';
  disabled?: boolean;
}

export const ROBOT_TYPES: Record<string, RobotType> = {
  BIPED: {
    id: 'biped',
    name: 'Biped Boss',
    description: 'Two-legged humanoid robot',
    sensorCount: 28,
    motorCount: 8,
    modelSize: 'small'
  },
  QUADRUPED: {
    id: 'quadruped',
    name: 'Quad Boss',
    description: 'Four-legged robot',
    sensorCount: 32,
    motorCount: 12,
    modelSize: 'medium',
    disabled: false
  },
  SPIDER: {
    id: 'spider',
    name: 'Spider Boss',
    description: 'Six-legged robot',
    sensorCount: 40,
    motorCount: 18,
    modelSize: 'large',
    disabled: false
  }
} as const;