import { OrbitControls, Grid, Box } from '@react-three/drei'
import { Physics, RigidBody } from '@react-three/rapier'
import { AIBoss } from './AIBoss'
import { AIQuadBoss } from './AIQuadBoss'
import { AISpiderBoss } from './AISpiderBoss'
import { CanvasCapture } from './CanvasCapture'
import { BossController } from './BossController'
import { RobotType } from '@/lib/robotTypes'

interface SceneProps {
  resetKey: number;
  controller: BossController;
  onStateUpdate: (sensorData: number[]) => void;
  onSensorUpdate: (data: { sensors: number[]; groundContact: any; fitness: number }) => void;
  onCapture: (pixels: Uint8Array) => void;
  isCapturing: boolean;
  simulationSpeed: number;
  robotType: RobotType;
}

// Main scene with enhanced AI and sensor feedback
export function Scene({ resetKey, controller, onStateUpdate, onSensorUpdate, onCapture, isCapturing, simulationSpeed, robotType }: SceneProps) {
  return (
    <Physics 
      gravity={[0, -9.81, 0]} 
      debug={false}
      timeStep={1/60 * simulationSpeed}
      maxVelocityIterations={Math.ceil(4 * simulationSpeed)}
      maxVelocityFrictionIterations={Math.ceil(1 * simulationSpeed)}
      maxStabilizationIterations={Math.ceil(1 * simulationSpeed)}
    >
      <ambientLight intensity={0.4} />
      <directionalLight 
        position={[5, 8, 5]} 
        intensity={0.8}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight position={[-5, 5, -5]} intensity={0.3} />
      
      {/* Ground - Solid floor with proper physics material */}
      <RigidBody 
        type="fixed" 
        position={[0, -2, 0]}
        friction={1.5}
        restitution={0.1}
      >
        <Box args={[50, 0.2, 50]}>
          <meshStandardMaterial color="#333" />
        </Box>
      </RigidBody>
      
      {/* Training area boundary indicators */}
      <group>
        {/* Corner posts */}
        <Box position={[15, 0, 15]} args={[0.2, 4, 0.2]}>
          <meshStandardMaterial color="#ff6b6b" transparent opacity={0.7} />
        </Box>
        <Box position={[-15, 0, 15]} args={[0.2, 4, 0.2]}>
          <meshStandardMaterial color="#ff6b6b" transparent opacity={0.7} />
        </Box>
        <Box position={[15, 0, -15]} args={[0.2, 4, 0.2]}>
          <meshStandardMaterial color="#ff6b6b" transparent opacity={0.7} />
        </Box>
        <Box position={[-15, 0, -15]} args={[0.2, 4, 0.2]}>
          <meshStandardMaterial color="#ff6b6b" transparent opacity={0.7} />
        </Box>
      </group>
      
      {/* Enhanced AI Boss with full sensor suite */}
      {robotType.id === 'biped' && (
        <AIBoss 
          key={resetKey} 
          controller={controller} 
          onStateUpdate={onStateUpdate}
          onSensorUpdate={onSensorUpdate}
          simulationSpeed={simulationSpeed}
          onBoundaryExit={() => {
            // Trigger a reset by incrementing the resetKey
            window.dispatchEvent(new CustomEvent('ragdollBoundaryExit'))
          }}
        />
      )}
      
      {robotType.id === 'quadruped' && (
        <AIQuadBoss 
          key={resetKey} 
          controller={controller} 
          onStateUpdate={onStateUpdate}
          onSensorUpdate={onSensorUpdate}
          simulationSpeed={simulationSpeed}
          onBoundaryExit={() => {
            // Trigger a reset by incrementing the resetKey
            window.dispatchEvent(new CustomEvent('ragdollBoundaryExit'))
          }}
        />
      )}
      
      {robotType.id === 'spider' && (
        <AISpiderBoss 
          key={resetKey} 
          controller={controller} 
          onStateUpdate={onStateUpdate}
          onSensorUpdate={onSensorUpdate}
          simulationSpeed={simulationSpeed}
          onBoundaryExit={() => {
            // Trigger a reset by incrementing the resetKey
            window.dispatchEvent(new CustomEvent('ragdollBoundaryExit'))
          }}
        />
      )}
      
      {/* Canvas capture for computer vision */}
      <CanvasCapture onCapture={onCapture} isCapturing={isCapturing} />
      
      <OrbitControls
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        minDistance={3}
        maxDistance={25}
        target={[0, 1, 0]}
        enableDamping={true}
        dampingFactor={0.05}
      />
    </Physics>
  )
}