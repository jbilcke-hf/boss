import { useRef, useState, useEffect, MutableRefObject } from 'react'
import { useFrame } from '@react-three/fiber'
import { Box, Sphere } from '@react-three/drei'
import { RigidBody, useSphericalJoint } from '@react-three/rapier'
import { BossController } from './BossController'

interface SpiderBossRefs {
  head: MutableRefObject<any>;
  torso: MutableRefObject<any>;
  frontLeftThigh: MutableRefObject<any>;
  frontRightThigh: MutableRefObject<any>;
  frontLeftShin: MutableRefObject<any>;
  frontRightShin: MutableRefObject<any>;
  frontLeftFoot: MutableRefObject<any>;
  frontRightFoot: MutableRefObject<any>;
  midLeftThigh: MutableRefObject<any>;
  midRightThigh: MutableRefObject<any>;
  midLeftShin: MutableRefObject<any>;
  midRightShin: MutableRefObject<any>;
  midLeftFoot: MutableRefObject<any>;
  midRightFoot: MutableRefObject<any>;
  backLeftThigh: MutableRefObject<any>;
  backRightThigh: MutableRefObject<any>;
  backLeftShin: MutableRefObject<any>;
  backRightShin: MutableRefObject<any>;
  backLeftFoot: MutableRefObject<any>;
  backRightFoot: MutableRefObject<any>;
}

interface AISpiderBossProps {
  controller: BossController;
  onStateUpdate: (sensorData: number[]) => void;
  onSensorUpdate: (data: { sensors: number[]; groundContact: any; fitness: number }) => void;
  onBoundaryExit?: () => void;
  simulationSpeed: number;
}

export function AISpiderBoss({ controller, onStateUpdate, onSensorUpdate, onBoundaryExit, simulationSpeed }: AISpiderBossProps) {
  const bossRefs: SpiderBossRefs = {
    head: useRef(null),
    torso: useRef(null),
    frontLeftThigh: useRef(null),
    frontRightThigh: useRef(null),
    frontLeftShin: useRef(null),
    frontRightShin: useRef(null),
    frontLeftFoot: useRef(null),
    frontRightFoot: useRef(null),
    midLeftThigh: useRef(null),
    midRightThigh: useRef(null),
    midLeftShin: useRef(null),
    midRightShin: useRef(null),
    midLeftFoot: useRef(null),
    midRightFoot: useRef(null),
    backLeftThigh: useRef(null),
    backRightThigh: useRef(null),
    backLeftShin: useRef(null),
    backRightShin: useRef(null),
    backLeftFoot: useRef(null),
    backRightFoot: useRef(null)
  }

  const [currentAction, setCurrentAction] = useState([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
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
  
  const BOUNDARY_LIMITS = {
    x: 15,
    y: -5,
    z: 15
  }

  // Joint connections for spider structure - 6 legs
  useSphericalJoint(bossRefs.head, bossRefs.torso, [[0, -0.08, 0], [0, 0.2, 0.35]])
  
  // Front legs (2)
  useSphericalJoint(bossRefs.frontLeftThigh, bossRefs.torso, [[0, 0.12, 0], [-0.25, -0.1, 0.25]])
  useSphericalJoint(bossRefs.frontLeftShin, bossRefs.frontLeftThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.frontLeftFoot, bossRefs.frontLeftShin, [[0, 0.02, -0.02], [0, -0.12, 0]])
  
  useSphericalJoint(bossRefs.frontRightThigh, bossRefs.torso, [[0, 0.12, 0], [0.25, -0.1, 0.25]])
  useSphericalJoint(bossRefs.frontRightShin, bossRefs.frontRightThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.frontRightFoot, bossRefs.frontRightShin, [[0, 0.02, -0.02], [0, -0.12, 0]])
  
  // Middle legs (2)
  useSphericalJoint(bossRefs.midLeftThigh, bossRefs.torso, [[0, 0.12, 0], [-0.3, -0.1, 0]])
  useSphericalJoint(bossRefs.midLeftShin, bossRefs.midLeftThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.midLeftFoot, bossRefs.midLeftShin, [[0, 0.02, -0.02], [0, -0.12, 0]])
  
  useSphericalJoint(bossRefs.midRightThigh, bossRefs.torso, [[0, 0.12, 0], [0.3, -0.1, 0]])
  useSphericalJoint(bossRefs.midRightShin, bossRefs.midRightThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.midRightFoot, bossRefs.midRightShin, [[0, 0.02, -0.02], [0, -0.12, 0]])
  
  // Back legs (2)
  useSphericalJoint(bossRefs.backLeftThigh, bossRefs.torso, [[0, 0.12, 0], [-0.25, -0.1, -0.25]])
  useSphericalJoint(bossRefs.backLeftShin, bossRefs.backLeftThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.backLeftFoot, bossRefs.backLeftShin, [[0, 0.02, -0.02], [0, -0.12, 0]])
  
  useSphericalJoint(bossRefs.backRightThigh, bossRefs.torso, [[0, 0.12, 0], [0.25, -0.1, -0.25]])
  useSphericalJoint(bossRefs.backRightShin, bossRefs.backRightThigh, [[0, 0.15, 0], [0, -0.15, 0]])
  useSphericalJoint(bossRefs.backRightFoot, bossRefs.backRightShin, [[0, 0.02, -0.02], [0, -0.12, 0]])

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
    if (!bossRefs.torso?.current || !bossRefs.head?.current) return

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

    if (!initializationDone.current) {
      if (!controller.isInitialized) {
        controller.createModel()
      }
      initializationDone.current = true
    }
    const isSettled = true

    const episodeTimeout = 3.0 / simulationSpeed
    if (episodeTimer.current > episodeTimeout) {
      const avgEpisodeFitness = episodeSamples.current > 0 ? episodeFitnessSum.current / episodeSamples.current : 0
      episodeCount.current++
      console.log(`Spider Episode ${episodeCount.current} complete - Avg fitness: ${avgEpisodeFitness.toFixed(2)}`)
      
      if (episodeStartState.current && episodeActions.current.length > 0 && controller.isInitialized && controller.trainingActive) {
        const finalAction = episodeActions.current[episodeActions.current.length - 1] || new Array(18).fill(0)
        controller.addTrainingSample(episodeStartState.current, finalAction, avgEpisodeFitness)
        console.log(`Spider Training sample added for episode ${episodeCount.current} - Actions: ${episodeActions.current.length}, Fitness: ${avgEpisodeFitness.toFixed(2)}`)
      }
      
      episodeTimer.current = 0
      episodeFitnessSum.current = 0
      episodeSamples.current = 0
      episodeStartState.current = null
      episodeActions.current = []
      episodeStarted.current = false
      
      onBoundaryExit?.()
      return
    }

    const trainingTimeout = 10.0 / simulationSpeed
    if (trainingTimer.current > trainingTimeout && !controller.isTraining && controller.trainingActive && controller.trainingData.length >= 3) {
      controller.trainModel()
      trainingTimer.current = 0
    }

    if (actionTimer.current > 0.05 && isSettled) {
      const sensorData = controller.getSensorData(bossRefs, actualDelta)
      
      if (sensorData) {
        const fitness = controller.calculateFitness(sensorData)
        
        if (!episodeStarted.current && controller.trainingActive) {
          episodeStartState.current = sensorData.slice(0, 40)
          episodeStarted.current = true
          episodeActions.current = []
        }
        
        episodeFitnessSum.current += fitness
        episodeSamples.current++
        
        onStateUpdate(sensorData)
        onSensorUpdate({
          sensors: sensorData.slice(0, 40),
          groundContact: (sensorData as any).groundContact,
          fitness: fitness
        })
      }

      if (sensorData && controller.isInitialized) {
        const action = await controller.predict(sensorData)
        setCurrentAction(action)

        if (controller.trainingActive) {
          episodeActions.current.push([...action])
        }

        lastSensorData.current = sensorData as any
      }
      
      actionTimer.current = 0
    }

    // Apply 18-motor control system for spider (6 legs × 3 joints each)
    if (bossRefs.torso.current && currentAction.length >= 18 && isSettled) {
      const [frontLeftHip, frontRightHip, frontLeftKnee, frontRightKnee, frontLeftAnkle, frontRightAnkle,
             midLeftHip, midRightHip, midLeftKnee, midRightKnee, midLeftAnkle, midRightAnkle,
             backLeftHip, backRightHip, backLeftKnee, backRightKnee, backLeftAnkle, backRightAnkle] = currentAction
      
      try {
        const motorStrength = 0.003 // Half the previous speed (0.006 / 2)
        
        // Front legs
        if (bossRefs.frontLeftThigh?.current && Math.abs(frontLeftHip) > 0.02) {
          bossRefs.frontLeftThigh.current.addTorque({
            x: frontLeftHip * motorStrength * 0.3,
            y: 0,
            z: frontLeftHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.frontLeftShin?.current && Math.abs(frontLeftKnee) > 0.02) {
          bossRefs.frontLeftShin.current.addTorque({
            x: frontLeftKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.frontLeftFoot?.current && Math.abs(frontLeftAnkle) > 0.02) {
          bossRefs.frontLeftFoot.current.addTorque({
            x: frontLeftAnkle * motorStrength * 0.15,
            y: 0,
            z: frontLeftAnkle * motorStrength * 0.08
          }, true)
        }
        
        if (bossRefs.frontRightThigh?.current && Math.abs(frontRightHip) > 0.02) {
          bossRefs.frontRightThigh.current.addTorque({
            x: frontRightHip * motorStrength * 0.3,
            y: 0,
            z: frontRightHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.frontRightShin?.current && Math.abs(frontRightKnee) > 0.02) {
          bossRefs.frontRightShin.current.addTorque({
            x: frontRightKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.frontRightFoot?.current && Math.abs(frontRightAnkle) > 0.02) {
          bossRefs.frontRightFoot.current.addTorque({
            x: frontRightAnkle * motorStrength * 0.15,
            y: 0,
            z: frontRightAnkle * motorStrength * 0.08
          }, true)
        }
        
        // Middle legs
        if (bossRefs.midLeftThigh?.current && Math.abs(midLeftHip) > 0.02) {
          bossRefs.midLeftThigh.current.addTorque({
            x: midLeftHip * motorStrength * 0.3,
            y: 0,
            z: midLeftHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.midLeftShin?.current && Math.abs(midLeftKnee) > 0.02) {
          bossRefs.midLeftShin.current.addTorque({
            x: midLeftKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.midLeftFoot?.current && Math.abs(midLeftAnkle) > 0.02) {
          bossRefs.midLeftFoot.current.addTorque({
            x: midLeftAnkle * motorStrength * 0.15,
            y: 0,
            z: midLeftAnkle * motorStrength * 0.08
          }, true)
        }
        
        if (bossRefs.midRightThigh?.current && Math.abs(midRightHip) > 0.02) {
          bossRefs.midRightThigh.current.addTorque({
            x: midRightHip * motorStrength * 0.3,
            y: 0,
            z: midRightHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.midRightShin?.current && Math.abs(midRightKnee) > 0.02) {
          bossRefs.midRightShin.current.addTorque({
            x: midRightKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.midRightFoot?.current && Math.abs(midRightAnkle) > 0.02) {
          bossRefs.midRightFoot.current.addTorque({
            x: midRightAnkle * motorStrength * 0.15,
            y: 0,
            z: midRightAnkle * motorStrength * 0.08
          }, true)
        }
        
        // Back legs
        if (bossRefs.backLeftThigh?.current && Math.abs(backLeftHip) > 0.02) {
          bossRefs.backLeftThigh.current.addTorque({
            x: backLeftHip * motorStrength * 0.3,
            y: 0,
            z: backLeftHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.backLeftShin?.current && Math.abs(backLeftKnee) > 0.02) {
          bossRefs.backLeftShin.current.addTorque({
            x: backLeftKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.backLeftFoot?.current && Math.abs(backLeftAnkle) > 0.02) {
          bossRefs.backLeftFoot.current.addTorque({
            x: backLeftAnkle * motorStrength * 0.15,
            y: 0,
            z: backLeftAnkle * motorStrength * 0.08
          }, true)
        }
        
        if (bossRefs.backRightThigh?.current && Math.abs(backRightHip) > 0.02) {
          bossRefs.backRightThigh.current.addTorque({
            x: backRightHip * motorStrength * 0.3,
            y: 0,
            z: backRightHip * motorStrength * 0.15
          }, true)
        }
        
        if (bossRefs.backRightShin?.current && Math.abs(backRightKnee) > 0.02) {
          bossRefs.backRightShin.current.addTorque({
            x: backRightKnee * motorStrength * 0.25,
            y: 0,
            z: 0
          }, true)
        }
        
        if (bossRefs.backRightFoot?.current && Math.abs(backRightAnkle) > 0.02) {
          bossRefs.backRightFoot.current.addTorque({
            x: backRightAnkle * motorStrength * 0.15,
            y: 0,
            z: backRightAnkle * motorStrength * 0.08
          }, true)
        }
        
      } catch (error) {
        console.warn('Spider Physics update error:', error)
      }
    }
  })

  return (
    <group position={[0, 0, 0]}>
      {/* Head */}
      <RigidBody 
        ref={bossRefs.head} 
        mass={0.25} 
        position={[0, 0.3, 0.35]}
        friction={0.8}
        restitution={0.1}
      >
        <Sphere args={[0.12]}>
          <meshStandardMaterial color="#f39c12" />
        </Sphere>
      </RigidBody>
      
      {/* Torso - spider-like body */}
      <RigidBody 
        ref={bossRefs.torso} 
        mass={2.5} 
        position={[0, 0, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.6, 0.25, 0.7]}>
          <meshStandardMaterial color="#2c3e50" />
        </Box>
      </RigidBody>

      {/* Front Left Leg */}
      <RigidBody 
        ref={bossRefs.frontLeftThigh} 
        mass={0.6} 
        position={[-0.25, -0.18, 0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.frontLeftShin} 
        mass={0.4} 
        position={[-0.25, -0.42, 0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.frontLeftFoot} 
        mass={0.15} 
        position={[-0.25, -0.58, 0.23]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Front Right Leg */}
      <RigidBody 
        ref={bossRefs.frontRightThigh} 
        mass={0.6} 
        position={[0.25, -0.18, 0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.frontRightShin} 
        mass={0.4} 
        position={[0.25, -0.42, 0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#27ae60" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.frontRightFoot} 
        mass={0.15} 
        position={[0.25, -0.58, 0.23]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Middle Left Leg */}
      <RigidBody 
        ref={bossRefs.midLeftThigh} 
        mass={0.6} 
        position={[-0.3, -0.18, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#3498db" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.midLeftShin} 
        mass={0.4} 
        position={[-0.3, -0.42, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#3498db" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.midLeftFoot} 
        mass={0.15} 
        position={[-0.3, -0.58, -0.02]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Middle Right Leg */}
      <RigidBody 
        ref={bossRefs.midRightThigh} 
        mass={0.6} 
        position={[0.3, -0.18, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#3498db" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.midRightShin} 
        mass={0.4} 
        position={[0.3, -0.42, 0]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#3498db" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.midRightFoot} 
        mass={0.15} 
        position={[0.3, -0.58, -0.02]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Back Left Leg */}
      <RigidBody 
        ref={bossRefs.backLeftThigh} 
        mass={0.6} 
        position={[-0.25, -0.18, -0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#e74c3c" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.backLeftShin} 
        mass={0.4} 
        position={[-0.25, -0.42, -0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#e74c3c" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.backLeftFoot} 
        mass={0.15} 
        position={[-0.25, -0.58, -0.23]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

      {/* Back Right Leg */}
      <RigidBody 
        ref={bossRefs.backRightThigh} 
        mass={0.6} 
        position={[0.25, -0.18, -0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.06, 0.25, 0.06]}>
          <meshStandardMaterial color="#e74c3c" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.backRightShin} 
        mass={0.4} 
        position={[0.25, -0.42, -0.25]}
        friction={0.8}
        restitution={0.1}
      >
        <Box args={[0.05, 0.24, 0.05]}>
          <meshStandardMaterial color="#e74c3c" />
        </Box>
      </RigidBody>
      
      <RigidBody 
        ref={bossRefs.backRightFoot} 
        mass={0.15} 
        position={[0.25, -0.58, -0.23]}
        friction={1.2}
        restitution={0.05}
      >
        <Box args={[0.08, 0.03, 0.12]}>
          <meshStandardMaterial color="#8e44ad" />
        </Box>
      </RigidBody>

    </group>
  )
}