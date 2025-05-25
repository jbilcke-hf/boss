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
}

// AI-controlled Boss with comprehensive sensor feedback
export function AIBoss({ controller, onStateUpdate, onSensorUpdate, onBoundaryExit }: AIBossProps) {
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
  
  // Boundary limits for training area
  const BOUNDARY_LIMITS = {
    x: 15,  // ±15 units in X
    y: -5,  // Below -5 units in Y (underground)
    z: 15   // ±15 units in Z
  }

  // Physics Joints
  useSphericalJoint(bossRefs.head, bossRefs.torso, [[0, -0.2, 0], [0, 0.4, 0]])
  useSphericalJoint(bossRefs.leftThigh, bossRefs.torso, [[0, 0.25, 0], [-0.15, -0.4, 0]])
  useSphericalJoint(bossRefs.leftShin, bossRefs.leftThigh, [[0, 0.2, 0], [0, -0.25, 0]])
  useSphericalJoint(bossRefs.leftFoot, bossRefs.leftShin, [[0, 0.025, -0.05], [0, -0.2, 0]])
  useSphericalJoint(bossRefs.rightThigh, bossRefs.torso, [[0, 0.25, 0], [0.15, -0.4, 0]])
  useSphericalJoint(bossRefs.rightShin, bossRefs.rightThigh, [[0, 0.2, 0], [0, -0.25, 0]])
  useSphericalJoint(bossRefs.rightFoot, bossRefs.rightShin, [[0, 0.025, -0.05], [0, -0.2, 0]])

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

    // Auto-reset episode every 3 seconds for rapid learning
    if (episodeTimer.current > 3.0) {
      const avgEpisodeFitness = episodeSamples.current > 0 ? episodeFitnessSum.current / episodeSamples.current : 0
      episodeCount.current++
      console.log(`Episode ${episodeCount.current} complete - Avg fitness: ${avgEpisodeFitness.toFixed(2)}`)
      
      // Reset episode tracking
      episodeTimer.current = 0
      episodeFitnessSum.current = 0
      episodeSamples.current = 0
      
      // Trigger reset via boundary exit mechanism
      onBoundaryExit?.()
      return
    }

    // Train model every 15 seconds (after several episodes)
    if (trainingTimer.current > 15.0 && !controller.isTraining) {
      controller.trainModel()
      trainingTimer.current = 0
    }

    // Update AI every 0.05 seconds (20Hz) for more responsive control
    if (actionTimer.current > 0.05 && isSettled) {
      const sensorData = controller.getSensorData(bossRefs, actualDelta)
      
      if (sensorData) {
        const fitness = controller.calculateFitness(sensorData)
        
        // Track episode fitness
        episodeFitnessSum.current += fitness
        episodeSamples.current++
        
        onStateUpdate(sensorData)
        onSensorUpdate({
          sensors: sensorData.slice(0, 24),
          groundContact: (sensorData as any).groundContact,
          fitness: fitness
        })
      }

      if (sensorData && controller.isInitialized) {
        const action = await controller.predict(sensorData)
        setCurrentAction(action)

        // Calculate fitness and add training sample
        if (lastSensorData.current) {
          const fitness = controller.calculateFitness(sensorData)
          controller.addTrainingSample((lastSensorData.current as any).slice(0, 24), currentAction, fitness)
        }

        lastSensorData.current = sensorData as any
      }
      
      actionTimer.current = 0
    }

    // Apply gentle AI motor controls to Boss (only after settling)
    if (bossRefs.torso.current && currentAction.length >= 8 && isSettled) {
      const [torqueX, torqueY, torqueZ, forceStrength, 
             leftLegTorque, rightLegTorque, balanceX, balanceZ] = currentAction
      
      try {
        // DISABLED: All artificial forces causing unnatural physics
        // The upward Y force was creating "reverse gravity" effect
        // Making the biped bounce to the sky after hitting ground
        /*
        const mainForce = {
          x: 0,
          y: Math.max(0, torqueY * forceStrength * 0.2), // This was the culprit!
          z: 0
        }
        
        if (bossRefs.torso.current && typeof bossRefs.torso.current.addForce === 'function') {
          bossRefs.torso.current.addForce(mainForce, true)
        }
        */

        // DISABLED: Balance torques also causing unnatural motion
        // These torques were contributing to the "wind" effect
        /*
        const balanceTorque = {
          x: balanceX * 0.1,
          y: 0,
          z: balanceZ * 0.1
        }
        
        if (bossRefs.torso.current && typeof bossRefs.torso.current.addTorque === 'function') {
          bossRefs.torso.current.addTorque(balanceTorque, true)
        }
        */

        // DISABLED: Leg motor controls causing helicopter effect
        // These horizontal forces on thighs were creating unnatural propulsion
        // and making the biped "fly" like helicopter blades
        /* 
        if (bossRefs.leftThigh?.current && Math.abs(leftLegTorque) > 0.05 && 
            typeof bossRefs.leftThigh.current.addForce === 'function') {
          const leftLegForce = {
            x: leftLegTorque * 0.2,
            y: 0,
            z: 0
          }
          bossRefs.leftThigh.current.addForce(leftLegForce, true)
        }

        if (bossRefs.rightThigh?.current && Math.abs(rightLegTorque) > 0.05 && 
            typeof bossRefs.rightThigh.current.addForce === 'function') {
          const rightLegForce = {
            x: rightLegTorque * 0.2,
            y: 0,
            z: 0
          }
          bossRefs.rightThigh.current.addForce(rightLegForce, true)
        }
        */
      } catch (error) {
        // Silently handle physics errors to prevent crashes
        console.warn('Physics update error:', error)
      }
    }
  })

  return (
    <group position={[0, 0, 0]}>
      {/* Head */}
      <RigidBody 
        ref={bossRefs.head} 
        mass={0.5} 
        position={[0, 0.6, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Sphere args={[0.2]}>
          <meshStandardMaterial color="#f39c12" />
        </Sphere>
      </RigidBody>
      
      {/* Torso - Main control point */}
      <RigidBody 
        ref={bossRefs.torso} 
        mass={2} 
        position={[0, 0, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.4, 0.8, 0.2]}>
          <meshStandardMaterial color="#4a90e2" />
        </Box>
      </RigidBody>

      {/* Left Leg */}
      <RigidBody 
        ref={bossRefs.leftThigh} 
        mass={1.2} 
        position={[-0.15, -0.3, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.12, 0.5, 0.12]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.leftShin} 
        mass={0.8} 
        position={[-0.15, -0.85, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.1, 0.4, 0.1]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.leftFoot} 
        mass={0.4} 
        position={[-0.15, -1.15, 0.05]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.15, 0.05, 0.25]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Right Leg */}
      <RigidBody 
        ref={bossRefs.rightThigh} 
        mass={1.2} 
        position={[0.15, -0.3, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.12, 0.5, 0.12]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.rightShin} 
        mass={0.8} 
        position={[0.15, -0.85, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.1, 0.4, 0.1]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.rightFoot} 
        mass={0.4} 
        position={[0.15, -1.15, 0.05]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.15, 0.05, 0.25]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

    </group>
  )
}