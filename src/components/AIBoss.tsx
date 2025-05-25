import { useRef, useState, useEffect, MutableRefObject } from 'react'
import { useFrame } from '@react-three/fiber'
import { Box, Sphere } from '@react-three/drei'
import { RigidBody, useSphericalJoint } from '@react-three/rapier'
import { BossController } from './BossController'

interface BossRefs {
  head: MutableRefObject<any>;
  torso: MutableRefObject<any>;
  leftThigh: MutableRefObject<any>;
  rightThigh: MutableRefObject<any>;
  leftShin: MutableRefObject<any>;
  rightShin: MutableRefObject<any>;
  leftFoot: MutableRefObject<any>;
  rightFoot: MutableRefObject<any>;
}

interface AIBossProps {
  controller: BossController;
  onStateUpdate: (sensorData: number[]) => void;
  onSensorUpdate: (data: { sensors: number[]; groundContact: any; fitness: number }) => void;
  onBoundaryExit?: () => void;
  simulationSpeed: number;
}

// AI-controlled Boss with comprehensive sensor feedback
export function AIBoss({ controller, onStateUpdate, onSensorUpdate, onBoundaryExit, simulationSpeed }: AIBossProps) {
  const bossRefs: BossRefs = {
    head: useRef(null),
    torso: useRef(null),
    leftThigh: useRef(null),
    rightThigh: useRef(null),
    leftShin: useRef(null),
    rightShin: useRef(null),
    leftFoot: useRef(null),
    rightFoot: useRef(null)
  }

  const [currentAction, setCurrentAction] = useState([0, 0, 0, 0, 0, 0, 0, 0])
  const lastSensorData = useRef(null)
  const actionTimer = useRef(0)
  const lastFrameTime = useRef(performance.now())
  const initializationDone = useRef(false)
  const trainingTimer = useRef(0)
  const episodeTimer = useRef(0)
  const episodeCount = useRef(0)
  const episodeFitnessSum = useRef(0)
  const episodeSamples = useRef(0)
  const episodeStartState = useRef<number[] | null>(null)
  const episodeActions = useRef<number[][]>([])
  const episodeStarted = useRef(false)
  
  // Boundary limits for training area
  const BOUNDARY_LIMITS = {
    x: 15,  // ±15 units in X
    y: -5,  // Below -5 units in Y (underground)
    z: 15   // ±15 units in Z
  }

  // More human-like joint positioning and constraints
  useSphericalJoint(bossRefs.head, bossRefs.torso, [[0, -0.15, 0], [0, 0.35, 0]])
  useSphericalJoint(bossRefs.leftThigh, bossRefs.torso, [[0, 0.2, 0], [-0.12, -0.35, 0]])
  useSphericalJoint(bossRefs.leftShin, bossRefs.leftThigh, [[0, 0.22, 0], [0, -0.22, 0]])
  useSphericalJoint(bossRefs.leftFoot, bossRefs.leftShin, [[0, 0.02, -0.03], [0, -0.18, 0]])
  useSphericalJoint(bossRefs.rightThigh, bossRefs.torso, [[0, 0.2, 0], [0.12, -0.35, 0]])
  useSphericalJoint(bossRefs.rightShin, bossRefs.rightThigh, [[0, 0.22, 0], [0, -0.22, 0]])
  useSphericalJoint(bossRefs.rightFoot, bossRefs.rightShin, [[0, 0.02, -0.03], [0, -0.18, 0]])

  // Check if ragdoll is within boundaries
  const checkBoundaries = () => {
    if (!bossRefs.torso?.current) return true
    
    const torsoPos = bossRefs.torso.current.translation()
    
    const isWithinBounds = 
      Math.abs(torsoPos.x) <= BOUNDARY_LIMITS.x &&
      torsoPos.y >= BOUNDARY_LIMITS.y &&
      Math.abs(torsoPos.z) <= BOUNDARY_LIMITS.z
    
    return isWithinBounds
  }

  useFrame(async (state, delta) => {
    // Only run frame updates when core refs are available
    if (!bossRefs.torso?.current || !bossRefs.head?.current) return

    // Check boundaries and trigger reset if needed
    if (!checkBoundaries()) {
      onBoundaryExit?.()
      return
    }

    actionTimer.current += delta
    trainingTimer.current += delta
    episodeTimer.current += delta
    const currentTime = performance.now()
    const actualDelta = (currentTime - lastFrameTime.current) / 1000
    lastFrameTime.current = currentTime

    // Initialize AI immediately and start balancing
    if (!initializationDone.current) {
      if (!controller.isInitialized) {
        controller.createModel()
      }
      initializationDone.current = true
    }
    const isSettled = true // Remove settling delay

    // Auto-reset episode every 3 seconds (scaled by simulation speed) for rapid learning  
    const episodeTimeout = 3.0 / simulationSpeed
    if (episodeTimer.current > episodeTimeout) {
      const avgEpisodeFitness = episodeSamples.current > 0 ? episodeFitnessSum.current / episodeSamples.current : 0
      episodeCount.current++
      console.log(`Episode ${episodeCount.current} complete - Avg fitness: ${avgEpisodeFitness.toFixed(2)}`)
      
      // Add single training sample for the entire episode (only if training is active)
      if (episodeStartState.current && episodeActions.current.length > 0 && controller.isInitialized && controller.trainingActive) {
        // Use the final action sequence and average fitness for this episode
        const finalAction = episodeActions.current[episodeActions.current.length - 1] || [0, 0, 0, 0, 0, 0, 0, 0]
        controller.addTrainingSample(episodeStartState.current, finalAction, avgEpisodeFitness)
        console.log(`Training sample added for episode ${episodeCount.current} - Actions: ${episodeActions.current.length}, Fitness: ${avgEpisodeFitness.toFixed(2)}`)
      }
      
      // Reset episode tracking
      episodeTimer.current = 0
      episodeFitnessSum.current = 0
      episodeSamples.current = 0
      episodeStartState.current = null
      episodeActions.current = []
      episodeStarted.current = false
      
      // Trigger reset via boundary exit mechanism
      onBoundaryExit?.()
      return
    }

    // Train model every 10 seconds (scaled by simulation speed) when training is active
    const trainingTimeout = 10.0 / simulationSpeed
    if (trainingTimer.current > trainingTimeout && !controller.isTraining && controller.trainingActive && controller.trainingData.length >= 3) {
      controller.trainModel()
      trainingTimer.current = 0
    }

    // Update AI every 0.05 seconds (20Hz) for more responsive control
    if (actionTimer.current > 0.05 && isSettled) {
      const sensorData = controller.getSensorData(bossRefs, actualDelta)
      
      if (sensorData) {
        const fitness = controller.calculateFitness(sensorData)
        
        // Capture initial state at start of episode (only if training is active)
        if (!episodeStarted.current && controller.trainingActive) {
          episodeStartState.current = sensorData.slice(0, 24)
          episodeStarted.current = true
          episodeActions.current = []
        }
        
        // Track episode fitness
        episodeFitnessSum.current += fitness
        episodeSamples.current++
        
        onStateUpdate(sensorData)
        onSensorUpdate({
          sensors: sensorData.slice(0, 28),
          groundContact: (sensorData as any).groundContact,
          fitness: fitness
        })
      }

      if (sensorData && controller.isInitialized) {
        const action = await controller.predict(sensorData)
        setCurrentAction(action)

        // Track actions during episode (only when training is active)
        if (controller.trainingActive) {
          episodeActions.current.push([...action])
        }

        lastSensorData.current = sensorData as any
      }
      
      actionTimer.current = 0
    }

    // Apply 8-motor control system with very gentle forces
    if (bossRefs.torso.current && currentAction.length >= 8 && isSettled) {
      const [leftHipTorque, rightHipTorque, leftKneeTorque, rightKneeTorque,
             leftAnkleTorque, rightAnkleTorque, torsoStabilize, headStabilize] = currentAction
      
      try {
        const motorStrength = 0.00625 // Half the previous speed (0.0125 / 2)
        
        // Hip motors (torso-thigh joints) - for stepping and balance
        if (bossRefs.leftThigh?.current && Math.abs(leftHipTorque) > 0.02) {
          const hipTorque = {
            x: leftHipTorque * motorStrength * 0.4,
            y: 0,
            z: leftHipTorque * motorStrength * 0.2
          }
          bossRefs.leftThigh.current.addTorque(hipTorque, true)
        }
        
        if (bossRefs.rightThigh?.current && Math.abs(rightHipTorque) > 0.02) {
          const hipTorque = {
            x: rightHipTorque * motorStrength * 0.4,
            y: 0,
            z: rightHipTorque * motorStrength * 0.2
          }
          bossRefs.rightThigh.current.addTorque(hipTorque, true)
        }
        
        // Knee motors (thigh-shin joints) - for leg bending
        if (bossRefs.leftShin?.current && Math.abs(leftKneeTorque) > 0.02) {
          const kneeTorque = {
            x: leftKneeTorque * motorStrength * 0.3,
            y: 0,
            z: 0
          }
          bossRefs.leftShin.current.addTorque(kneeTorque, true)
        }
        
        if (bossRefs.rightShin?.current && Math.abs(rightKneeTorque) > 0.02) {
          const kneeTorque = {
            x: rightKneeTorque * motorStrength * 0.3,
            y: 0,
            z: 0
          }
          bossRefs.rightShin.current.addTorque(kneeTorque, true)
        }
        
        // Ankle motors (shin-foot joints) - for foot positioning and balance
        if (bossRefs.leftFoot?.current && Math.abs(leftAnkleTorque) > 0.02) {
          const ankleTorque = {
            x: leftAnkleTorque * motorStrength * 0.2,
            y: 0,
            z: leftAnkleTorque * motorStrength * 0.1
          }
          bossRefs.leftFoot.current.addTorque(ankleTorque, true)
        }
        
        if (bossRefs.rightFoot?.current && Math.abs(rightAnkleTorque) > 0.02) {
          const ankleTorque = {
            x: rightAnkleTorque * motorStrength * 0.2,
            y: 0,
            z: rightAnkleTorque * motorStrength * 0.1
          }
          bossRefs.rightFoot.current.addTorque(ankleTorque, true)
        }
        
        // Torso stabilization motor - gentle balance correction
        if (bossRefs.torso?.current && Math.abs(torsoStabilize) > 0.02) {
          const stabTorque = {
            x: torsoStabilize * motorStrength * 0.1,
            y: 0,
            z: torsoStabilize * motorStrength * 0.1
          }
          bossRefs.torso.current.addTorque(stabTorque, true)
        }
        
        // Head stabilization motor - very gentle
        if (bossRefs.head?.current && Math.abs(headStabilize) > 0.02) {
          const headTorque = {
            x: headStabilize * motorStrength * 0.05,
            y: 0,
            z: headStabilize * motorStrength * 0.05
          }
          bossRefs.head.current.addTorque(headTorque, true)
        }
        
      } catch (error) {
        // Silently handle physics errors to prevent crashes
        console.warn('Physics update error:', error)
      }
    }
  })

  return (
    <group position={[0, 0, 0]}>
      {/* Head - more proportional */}
      <RigidBody 
        ref={bossRefs.head} 
        mass={0.4} 
        position={[0, 0.55, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Sphere args={[0.18]}>
          <meshStandardMaterial color="#f39c12" />
        </Sphere>
      </RigidBody>
      
      {/* Torso - more human proportions */}
      <RigidBody 
        ref={bossRefs.torso} 
        mass={2.5} 
        position={[0, 0, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.35, 0.7, 0.18]}>
          <meshStandardMaterial color="#4a90e2" />
        </Box>
      </RigidBody>

      {/* Left Leg - more human proportions */}
      <RigidBody 
        ref={bossRefs.leftThigh} 
        mass={1.0} 
        position={[-0.12, -0.28, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.11, 0.44, 0.11]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.leftShin} 
        mass={0.7} 
        position={[-0.12, -0.78, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.09, 0.36, 0.09]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.leftFoot} 
        mass={0.3} 
        position={[-0.12, -1.05, 0.03]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.13, 0.04, 0.22]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Right Leg - more human proportions */}
      <RigidBody 
        ref={bossRefs.rightThigh} 
        mass={1.0} 
        position={[0.12, -0.28, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.11, 0.44, 0.11]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.rightShin} 
        mass={0.7} 
        position={[0.12, -0.78, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.09, 0.36, 0.09]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.rightFoot} 
        mass={0.3} 
        position={[0.12, -1.05, 0.03]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.13, 0.04, 0.22]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

    </group>
  )
}