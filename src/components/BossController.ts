import * as tf from '@tensorflow/tfjs'
import { ModelExporter } from '@/lib/ModelExporter'
import { ROBOT_TYPES, type RobotType } from '@/lib/robotTypes'

interface TrainingSample {
  state: number[];
  action: number[];
  fitness: number;
}

interface PreviousState {
  headVel: { x: number; y: number; z: number };
  torsoVel: { x: number; y: number; z: number };
  timestamp: number;
}


// Advanced Neural Network for controlling Boss with full sensor data
export class BossController {
  model: tf.LayersModel | null = null;
  isTraining: boolean = false;
  trainingData: TrainingSample[] = [];
  isInitialized: boolean = false;
  previousState: PreviousState | null = null;
  sensorHistory: number[][] = [];
  robotType: RobotType;
  modelName: string;

  constructor(robotType: RobotType = ROBOT_TYPES.BIPED) {
    this.robotType = robotType
    this.modelName = `${robotType.name}_model_${Date.now()}`
  }

  // Create a neural network with comprehensive sensor inputs based on robot type
  createModel(): tf.LayersModel {
    const { sensorCount, motorCount } = this.robotType
    
    const model = tf.sequential({
      layers: [
        // Input layer sized for robot type
        tf.layers.dense({ inputShape: [sensorCount], units: 64, activation: 'relu', name: 'input_layer' }),
        tf.layers.dropout({ rate: 0.2, name: 'dropout_1' }),
        tf.layers.dense({ units: 32, activation: 'relu', name: 'hidden_1' }),
        tf.layers.dropout({ rate: 0.2, name: 'dropout_2' }),
        tf.layers.dense({ units: 16, activation: 'relu', name: 'hidden_2' }),
        // Output layer sized for robot type
        tf.layers.dense({ units: motorCount, activation: 'tanh', name: 'output_layer' })
      ]
    })

    model.compile({
      optimizer: tf.train.adam(0.0005),
      loss: 'meanSquaredError',
      metrics: ['mse']
    })

    this.model = model
    this.isInitialized = true
    console.log(`${this.robotType.name} AI Model created!`)
    console.log('Parameters:', model.countParams())
    console.log(`Sensor inputs: ${sensorCount}, Motor outputs: ${motorCount}`)
    return model
  }

  // Export model to Safetensors format
  async exportModel(): Promise<{ success: boolean; filename?: string; error?: string }> {
    if (!this.model) {
      throw new Error('No trained model to export')
    }

    try {
      const metadata = {
        robot_type: this.robotType.id,
        robot_name: this.robotType.name,
        sensor_count: this.robotType.sensorCount,
        motor_count: this.robotType.motorCount,
        training_samples: this.trainingData.length,
        model_name: this.modelName,
        export_version: '1.0'
      }

      const blob = await ModelExporter.exportToSafetensors(this.model, metadata)
      const filename = `${this.robotType.id}_boss_model_${new Date().toISOString().slice(0, 10)}.safetensors.json`
      
      ModelExporter.downloadBlob(blob, filename)
      console.log(`Model exported: ${filename}`)
      
      return { success: true, filename }
    } catch (error) {
      console.error('Export failed:', error)
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' }
    }
  }

  // Load model from uploaded file (placeholder for future implementation)
  async loadModel(file: File): Promise<{ success: boolean; error?: string }> {
    // TODO: Implement model loading from Safetensors format
    console.log('Model loading not yet implemented:', file.name)
    return { success: false, error: 'Loading not yet implemented' }
  }

  // Comprehensive sensor data collection
  getSensorData(bossRefs: any, deltaTime: number = 0.016): any {
    if (!bossRefs.head?.current || !bossRefs.torso?.current) return null
    
    const currentTime = performance.now()
    
    // Position sensors
    const headPos = bossRefs.head.current.translation()
    const torsoPos = bossRefs.torso.current.translation()
    const leftThighPos = bossRefs.leftThigh.current?.translation() || { x: 0, y: 0, z: 0 }
    const rightThighPos = bossRefs.rightThigh.current?.translation() || { x: 0, y: 0, z: 0 }
    const leftFootPos = bossRefs.leftFoot.current?.translation() || { x: 0, y: 0, z: 0 }
    const rightFootPos = bossRefs.rightFoot.current?.translation() || { x: 0, y: 0, z: 0 }
    
    // Rotation sensors (quaternions converted to euler-like)
    const torsoRot = bossRefs.torso.current.rotation()
    
    // Velocity sensors
    const headVel = bossRefs.head.current.linvel()
    const torsoVel = bossRefs.torso.current.linvel()
    
    // Angular velocity sensors
    const torsoAngVel = bossRefs.torso.current.angvel()

    // Calculate acceleration from velocity history
    let headAccel = { x: 0, y: 0, z: 0 }
    let torsoAccel = { x: 0, y: 0, z: 0 }
    
    if (this.previousState) {
      headAccel = {
        x: (headVel.x - this.previousState.headVel.x) / deltaTime,
        y: (headVel.y - this.previousState.headVel.y) / deltaTime,
        z: (headVel.z - this.previousState.headVel.z) / deltaTime
      }
      torsoAccel = {
        x: (torsoVel.x - this.previousState.torsoVel.x) / deltaTime,
        y: (torsoVel.y - this.previousState.torsoVel.y) / deltaTime,
        z: (torsoVel.z - this.previousState.torsoVel.z) / deltaTime
      }
    }

    // Calculate joint angles (simplified - distance-based approximation)
    const leftKneeAngle = Math.atan2(
      leftThighPos.y - (bossRefs.leftShin.current?.translation().y || 0),
      Math.abs(leftThighPos.x - (bossRefs.leftShin.current?.translation().x || 0))
    )
    const rightKneeAngle = Math.atan2(
      rightThighPos.y - (bossRefs.rightShin.current?.translation().y || 0),
      Math.abs(rightThighPos.x - (bossRefs.rightShin.current?.translation().x || 0))
    )

    // Center of mass calculation
    const centerOfMass = {
      x: (headPos.x + torsoPos.x + leftThighPos.x + rightThighPos.x) / 4,
      y: (headPos.y + torsoPos.y + leftThighPos.y + rightThighPos.y) / 4,
      z: (headPos.z + torsoPos.z + leftThighPos.z + rightThighPos.z) / 4
    }

    // Ground contact sensors (simplified)
    const leftFootGroundContact = leftFootPos.y < -1.5 ? 1.0 : 0.0
    const rightFootGroundContact = rightFootPos.y < -1.5 ? 1.0 : 0.0

    // 24-dimensional sensor vector
    const sensorData = [
      // Position sensors (6)
      headPos.y, torsoPos.y, 
      leftThighPos.y, rightThighPos.y,
      centerOfMass.y, torsoPos.x,
      
      // Velocity sensors (6)
      headVel.x, headVel.y, headVel.z,
      torsoVel.x, torsoVel.y, torsoVel.z,
      
      // Acceleration sensors (4)
      headAccel.y, torsoAccel.x, torsoAccel.y, torsoAccel.z,
      
      // Rotation/orientation sensors (4)
      torsoRot.x, torsoRot.y, torsoRot.z, torsoRot.w,
      
      // Angular velocity sensors (2)
      torsoAngVel.x, torsoAngVel.z,
      
      // Joint angle sensors (2)
      leftKneeAngle, rightKneeAngle
    ]

    // Store current state for next frame's acceleration calculation
    this.previousState = {
      headVel: { x: headVel.x, y: headVel.y, z: headVel.z },
      torsoVel: { x: torsoVel.x, y: torsoVel.y, z: torsoVel.z },
      timestamp: currentTime
    } as PreviousState

    // Add ground contact info to state
    (sensorData as any).groundContact = {
      left: leftFootGroundContact,
      right: rightFootGroundContact,
      both: leftFootGroundContact && rightFootGroundContact
    }

    return sensorData
  }

  // Get current state using comprehensive sensors
  getState(bossRefs: any, deltaTime: number): any {
    return this.getSensorData(bossRefs, deltaTime)
  }

  // Enhanced fitness function using sensor data - starts at perfect score, degrades as robot falls
  calculateFitness(sensorData: number[]): number {
    if (!sensorData || sensorData.length < 24) return 100 // Start with perfect score
    
    const [headY, torsoY, _leftThighY, _rightThighY, centerMassY, torsoX,
           _headVelX, _headVelY, _headVelZ, torsoVelX, torsoVelY, torsoVelZ,
           _headAccelY, torsoAccelX, torsoAccelY, torsoAccelZ,
           rotX, _rotY, rotZ, _rotW, angVelX, angVelZ,
           leftKneeAngle, rightKneeAngle] = sensorData

    // Start with perfect fitness and subtract penalties
    let fitness = 100.0

    // Target heights for optimal standing position
    const targetHeadHeight = 1.8   // Expected head height when standing
    const targetTorsoHeight = 1.0  // Expected torso height when standing
    
    // Height penalties - lose points as robot falls
    const headHeightPenalty = headY < targetHeadHeight * 0.8 ? 
      Math.min(30, (targetHeadHeight - headY) * 15) : 0
    
    const torsoHeightPenalty = torsoY < targetTorsoHeight * 0.7 ? 
      Math.min(25, (targetTorsoHeight - torsoY) * 20) : 0
    
    // Position stability penalty - lose points for drifting from center
    const positionDrift = Math.sqrt(torsoX * torsoX + (centerMassY - targetTorsoHeight) * (centerMassY - targetTorsoHeight))
    const positionPenalty = Math.min(15, positionDrift * 2)
    
    // Movement penalty - lose points for excessive motion
    const velocityMagnitude = Math.sqrt(torsoVelX * torsoVelX + torsoVelY * torsoVelY + torsoVelZ * torsoVelZ)
    const velocityPenalty = Math.min(10, Math.max(0, velocityMagnitude - 0.3) * 8)
    
    // Angular instability penalty - lose points for wobbling/rotating
    const angularPenalty = Math.min(15, (Math.abs(angVelX) + Math.abs(angVelZ)) * 3)
    
    // Orientation penalty - lose points for tilting
    const orientationPenalty = Math.min(15, (Math.abs(rotX) + Math.abs(rotZ)) * 25)
    
    // Ground contact penalty - heavy penalty for losing foot contact
    const groundContactPenalty = !(sensorData as any).groundContact?.both ? 
      (!(sensorData as any).groundContact?.left && !(sensorData as any).groundContact?.right ? 25 : 10) : 0
    
    // Joint angle penalty - lose points for unnatural poses
    const jointPenalty = Math.min(10, (Math.abs(leftKneeAngle) + Math.abs(rightKneeAngle)) * 2)
    
    // Apply all penalties
    fitness -= (headHeightPenalty + torsoHeightPenalty + positionPenalty + 
               velocityPenalty + angularPenalty + orientationPenalty + 
               groundContactPenalty + jointPenalty)

    return Math.max(0, fitness)
  }

  // Generate enhanced motor control actions
  async predict(sensorData: number[]): Promise<number[]> {
    if (!this.model || !sensorData || sensorData.length < this.robotType.sensorCount) {
      // Return default actions sized for robot type (all zeros for no forces)
      return new Array(this.robotType.motorCount).fill(0)
    }
    
    const stateTensor = tf.tensor2d([sensorData.slice(0, this.robotType.sensorCount)])
    const prediction = this.model.predict(stateTensor) as tf.Tensor
    const result = await prediction.data()
    
    stateTensor.dispose()
    prediction.dispose()
    
    return Array.from(result)
  }

  // Add training sample
  addTrainingSample(state: number[], action: number[], fitness: number): void {
    this.trainingData.push({ state, action, fitness })
    
    // Keep only recent samples
    if (this.trainingData.length > 1000) {
      this.trainingData = this.trainingData.slice(-800)
    }
  }

  // Train the model using collected data
  async trainModel(): Promise<void> {
    if (!this.model || this.trainingData.length < 50) return

    this.isTraining = true
    console.log('Training Boss AI with', this.trainingData.length, 'samples...')

    // Prepare training data
    const states = this.trainingData.map(sample => sample.state)
    const actions = this.trainingData.map(sample => sample.action)
    
    const xs = tf.tensor2d(states)
    const ys = tf.tensor2d(actions)

    try {
      await this.model.fit(xs, ys, {
        epochs: 10,
        batchSize: 32,
        shuffle: true,
        verbose: 0
      })
      console.log('Boss AI training complete!')
    } catch (error) {
      console.error('Training failed:', error)
    } finally {
      xs.dispose()
      ys.dispose()
      this.isTraining = false
    }
  }

  // Reset only the neural network model (not the training data)
  resetModel(): void {
    if (this.model) {
      this.model.dispose()
      this.model = null
    }
    this.createModel()
    console.log('Neural network model reset (training data preserved)')
  }

  // Reset only position-related state (for robot resets without losing training progress)
  resetPositionState(): void {
    this.previousState = null
    this.sensorHistory = []
    console.log('Robot position state reset (model and training data preserved)')
  }

  // Complete reset - clears everything (for when user wants to start over)
  resetAll(): void {
    if (this.model) {
      this.model.dispose()
      this.model = null
    }
    this.trainingData = []
    this.previousState = null
    this.sensorHistory = []
    this.isTraining = false
    this.isInitialized = false
    console.log('Complete reset - all data cleared')
  }
}