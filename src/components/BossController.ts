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
  explorationRate: number = 0.8; // Start with high exploration
  stepCount: number = 0;
  lastFitness: number = 0;
  fitnessHistory: number[] = [];
  bestAction: number[] | null = null;
  bestFitness: number = 0;

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

  // Enhanced fitness function optimized for 3-second episodes
  calculateFitness(sensorData: number[]): number {
    if (!sensorData || sensorData.length < 24) return 100 // Start with perfect score
    
    const [headY, torsoY, _leftThighY, _rightThighY, centerMassY, torsoX,
           _headVelX, _headVelY, _headVelZ, torsoVelX, torsoVelY, torsoVelZ,
           _headAccelY, torsoAccelX, torsoAccelY, torsoAccelZ,
           rotX, _rotY, rotZ, _rotW, angVelX, angVelZ,
           leftKneeAngle, rightKneeAngle] = sensorData

    // Base fitness for being upright - heavily weighted for short episodes
    let fitness = 0

    // Target heights for optimal standing position
    const targetHeadHeight = 1.8   // Expected head height when standing
    const targetTorsoHeight = 1.0  // Expected torso height when standing
    
    // Primary reward: Being upright (most important for short episodes)
    const headHeightScore = Math.max(0, 40 * (headY / targetHeadHeight))
    const torsoHeightScore = Math.max(0, 30 * (torsoY / targetTorsoHeight))
    
    // Stability bonus - reward staying in position
    const positionStability = Math.max(0, 15 * (1 - Math.min(1, Math.abs(torsoX) / 2)))
    
    // Balance bonus - reward low movement
    const velocityMagnitude = Math.sqrt(torsoVelX * torsoVelX + torsoVelY * torsoVelY + torsoVelZ * torsoVelZ)
    const stillnessBonus = Math.max(0, 10 * (1 - Math.min(1, velocityMagnitude / 2)))
    
    // Orientation bonus - reward staying vertical
    const uprightBonus = Math.max(0, 15 * (1 - Math.abs(rotX) - Math.abs(rotZ)))
    
    // Ground contact is essential - big bonus for having feet on ground
    const groundContactBonus = (sensorData as any).groundContact?.both ? 20 : 
                              ((sensorData as any).groundContact?.left || (sensorData as any).groundContact?.right) ? 10 : 0
    
    // Joint stability - reward natural poses
    const jointStabilityBonus = Math.max(0, 10 * (1 - (Math.abs(leftKneeAngle) + Math.abs(rightKneeAngle)) / 2))
    
    // Angular stability - reward minimal wobbling
    const angularStabilityBonus = Math.max(0, 10 * (1 - (Math.abs(angVelX) + Math.abs(angVelZ)) / 2))
    
    // Combine all fitness components - emphasize staying upright
    fitness = headHeightScore + torsoHeightScore + positionStability + 
              stillnessBonus + uprightBonus + groundContactBonus + 
              jointStabilityBonus + angularStabilityBonus

    return Math.max(0, Math.min(100, fitness)) // Clamp between 0-100
  }

  // Generate enhanced motor control actions with exploration
  async predict(sensorData: number[]): Promise<number[]> {
    this.stepCount++
    
    if (!this.model || !sensorData || sensorData.length < this.robotType.sensorCount) {
      // Return gentle random actions for exploration
      return this.generateExplorationAction()
    }
    
    // Decay exploration rate over time
    this.explorationRate = Math.max(0.1, this.explorationRate * 0.999)
    
    let action: number[]
    
    if (Math.random() < this.explorationRate) {
      // Exploration: try random actions
      action = this.generateExplorationAction()
    } else {
      // Exploitation: use neural network prediction
      const stateTensor = tf.tensor2d([sensorData.slice(0, this.robotType.sensorCount)])
      const prediction = this.model.predict(stateTensor) as tf.Tensor
      const result = await prediction.data()
      
      stateTensor.dispose()
      prediction.dispose()
      
      action = Array.from(result)
    }
    
    return action
  }
  
  // Generate gentle random exploration actions
  private generateExplorationAction(): number[] {
    const action = []
    for (let i = 0; i < this.robotType.motorCount; i++) {
      // Generate small random values for gentle exploration
      action.push((Math.random() - 0.5) * 0.5) // Range: -0.25 to 0.25
    }
    return action
  }

  // Add training sample with fitness tracking
  addTrainingSample(state: number[], action: number[], fitness: number): void {
    this.trainingData.push({ state, action, fitness })
    this.fitnessHistory.push(fitness)
    
    // Track best performance
    if (fitness > this.bestFitness) {
      this.bestFitness = fitness
      this.bestAction = [...action]
      console.log(`New best fitness: ${fitness.toFixed(2)}`)
    }
    
    // Keep only recent samples and history
    if (this.trainingData.length > 1000) {
      this.trainingData = this.trainingData.slice(-800)
    }
    if (this.fitnessHistory.length > 500) {
      this.fitnessHistory = this.fitnessHistory.slice(-400)
    }
    
    this.lastFitness = fitness
  }

  // Train the model using collected data with fitness-weighted approach
  async trainModel(): Promise<void> {
    if (!this.model || this.trainingData.length < 30) return

    this.isTraining = true
    console.log('Training Boss AI with', this.trainingData.length, 'samples...')
    
    // Filter and weight samples by fitness
    const sortedSamples = this.trainingData
      .filter(sample => sample.fitness > 20) // Only train on decent performance
      .sort((a, b) => b.fitness - a.fitness) // Best first
      .slice(0, 200) // Take best 200 samples
    
    if (sortedSamples.length < 10) {
      console.log('Not enough good samples for training yet')
      this.isTraining = false
      return
    }

    // Prepare training data - duplicate better samples
    const states: number[][] = []
    const actions: number[][] = []
    
    sortedSamples.forEach((sample, index) => {
      const weight = Math.max(1, Math.floor(sample.fitness / 25)) // Better samples get more weight
      for (let w = 0; w < weight; w++) {
        states.push(sample.state.slice(0, this.robotType.sensorCount))
        actions.push(sample.action)
      }
    })
    
    const xs = tf.tensor2d(states)
    const ys = tf.tensor2d(actions)

    try {
      await this.model.fit(xs, ys, {
        epochs: 5, // Reduced epochs for more frequent training
        batchSize: 16, // Smaller batch size
        shuffle: true,
        verbose: 0,
        validationSplit: 0.1
      })
      
      const avgFitness = this.fitnessHistory.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, this.fitnessHistory.length)
      console.log(`Training complete! Avg fitness: ${avgFitness.toFixed(2)}, Exploration: ${(this.explorationRate * 100).toFixed(1)}%`)
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
    // Don't reset exploration rate or step count - keep learning progress
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
    this.explorationRate = 0.8
    this.stepCount = 0
    this.lastFitness = 0
    this.fitnessHistory = []
    this.bestAction = null
    this.bestFitness = 0
    console.log('Complete reset - all data cleared')
  }
}