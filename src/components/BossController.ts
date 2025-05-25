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
  explorationRate: number = 0.7; // Start with high but reasonable exploration
  stepCount: number = 0;
  lastFitness: number = 0;
  fitnessHistory: number[] = [];
  bestAction: number[] | null = null;
  bestFitness: number = 0;
  trainingActive: boolean = false; // Whether training is currently active
  lastAction: number[] = [0, 0, 0, 0, 0, 0, 0, 0]; // For action smoothing
  maxMotorChangeRate: number = 0.05; // Much slower motor changes to prevent shaking

  constructor(robotType: RobotType = ROBOT_TYPES.BIPED) {
    this.robotType = robotType
    this.modelName = `${robotType.name}_model_${Date.now()}`
  }

  // Create a neural network with comprehensive sensor inputs based on robot type
  createModel(): tf.LayersModel {
    const { sensorCount, motorCount } = this.robotType
    
    const model = tf.sequential({
      layers: [
        // Simpler architecture for better exploration and learning
        tf.layers.dense({ inputShape: [sensorCount], units: 32, activation: 'relu', name: 'input_layer' }),
        tf.layers.dropout({ rate: 0.1, name: 'dropout_1' }),
        tf.layers.dense({ units: 16, activation: 'relu', name: 'hidden_1' }),
        tf.layers.dropout({ rate: 0.1, name: 'dropout_2' }),
        // Output layer with tanh activation for smooth motor control (-1 to 1)
        tf.layers.dense({ units: motorCount, activation: 'tanh', name: 'output_layer' })
      ]
    })

    model.compile({
      optimizer: tf.train.adam(0.001), // Higher learning rate for faster adaptation
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

  // Get limb positions based on robot type
  private getLimbPositions(bossRefs: any): any {
    const positions: any = {}
    
    if (this.robotType.id === 'biped') {
      positions.leftThigh = bossRefs.leftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.rightThigh = bossRefs.rightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.leftFoot = bossRefs.leftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.rightFoot = bossRefs.rightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
    } else if (this.robotType.id === 'quadruped') {
      positions.frontLeftThigh = bossRefs.frontLeftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontRightThigh = bossRefs.frontRightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backLeftThigh = bossRefs.backLeftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backRightThigh = bossRefs.backRightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontLeftFoot = bossRefs.frontLeftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontRightFoot = bossRefs.frontRightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backLeftFoot = bossRefs.backLeftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backRightFoot = bossRefs.backRightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
    } else if (this.robotType.id === 'spider') {
      positions.frontLeftThigh = bossRefs.frontLeftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontRightThigh = bossRefs.frontRightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.midLeftThigh = bossRefs.midLeftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.midRightThigh = bossRefs.midRightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backLeftThigh = bossRefs.backLeftThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backRightThigh = bossRefs.backRightThigh?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontLeftFoot = bossRefs.frontLeftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.frontRightFoot = bossRefs.frontRightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.midLeftFoot = bossRefs.midLeftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.midRightFoot = bossRefs.midRightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backLeftFoot = bossRefs.backLeftFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
      positions.backRightFoot = bossRefs.backRightFoot?.current?.translation() || { x: 0, y: 0, z: 0 }
    }
    
    return positions
  }

  // Create sensor data based on robot type
  private createSensorData(headPos: any, torsoPos: any, limbPositions: any, headVel: any, torsoVel: any, headAccel: any, torsoAccel: any, torsoRot: any, torsoAngVel: any, bossRefs: any): number[] {
    const groundLevel = -2.0
    const contactThreshold = 0.1
    
    // Common sensors for all robot types
    const commonSensors = [
      // Position sensors
      headPos.y, torsoPos.y, torsoPos.x,
      
      // Velocity sensors
      headVel.x, headVel.y, headVel.z,
      torsoVel.x, torsoVel.y, torsoVel.z,
      
      // Acceleration sensors
      headAccel.y, torsoAccel.x, torsoAccel.y, torsoAccel.z,
      
      // Rotation/orientation sensors
      torsoRot.x, torsoRot.y, torsoRot.z, torsoRot.w,
      
      // Angular velocity sensors
      torsoAngVel.x, torsoAngVel.z
    ]
    
    let sensorData = [...commonSensors]
    let groundContact: any = {}
    
    // Add robot-specific sensors
    if (this.robotType.id === 'biped') {
      const leftFootContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.leftFoot.y + contactThreshold) / contactThreshold))
      const rightFootContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.rightFoot.y + contactThreshold) / contactThreshold))
      
      sensorData.push(
        limbPositions.leftThigh.y, limbPositions.rightThigh.y,
        leftFootContact, rightFootContact,
        limbPositions.leftFoot.y, limbPositions.rightFoot.y
      )
      
      groundContact = { left: leftFootContact, right: rightFootContact, both: leftFootContact && rightFootContact }
      
    } else if (this.robotType.id === 'quadruped') {
      const frontLeftContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.frontLeftFoot.y + contactThreshold) / contactThreshold))
      const frontRightContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.frontRightFoot.y + contactThreshold) / contactThreshold))
      const backLeftContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.backLeftFoot.y + contactThreshold) / contactThreshold))
      const backRightContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.backRightFoot.y + contactThreshold) / contactThreshold))
      
      sensorData.push(
        limbPositions.frontLeftThigh.y, limbPositions.frontRightThigh.y,
        limbPositions.backLeftThigh.y, limbPositions.backRightThigh.y,
        frontLeftContact, frontRightContact, backLeftContact, backRightContact,
        limbPositions.frontLeftFoot.y, limbPositions.frontRightFoot.y,
        limbPositions.backLeftFoot.y, limbPositions.backRightFoot.y
      )
      
      groundContact = { 
        frontLeft: frontLeftContact, frontRight: frontRightContact,
        backLeft: backLeftContact, backRight: backRightContact,
        stable: (frontLeftContact + frontRightContact + backLeftContact + backRightContact) >= 3
      }
      
    } else if (this.robotType.id === 'spider') {
      const frontLeftContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.frontLeftFoot.y + contactThreshold) / contactThreshold))
      const frontRightContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.frontRightFoot.y + contactThreshold) / contactThreshold))
      const midLeftContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.midLeftFoot.y + contactThreshold) / contactThreshold))
      const midRightContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.midRightFoot.y + contactThreshold) / contactThreshold))
      const backLeftContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.backLeftFoot.y + contactThreshold) / contactThreshold))
      const backRightContact = Math.max(0, Math.min(1, (groundLevel - limbPositions.backRightFoot.y + contactThreshold) / contactThreshold))
      
      sensorData.push(
        limbPositions.frontLeftThigh.y, limbPositions.frontRightThigh.y,
        limbPositions.midLeftThigh.y, limbPositions.midRightThigh.y,
        limbPositions.backLeftThigh.y, limbPositions.backRightThigh.y,
        frontLeftContact, frontRightContact, midLeftContact, midRightContact, backLeftContact, backRightContact,
        limbPositions.frontLeftFoot.y, limbPositions.frontRightFoot.y,
        limbPositions.midLeftFoot.y, limbPositions.midRightFoot.y,
        limbPositions.backLeftFoot.y, limbPositions.backRightFoot.y
      )
      
      groundContact = { 
        frontLeft: frontLeftContact, frontRight: frontRightContact,
        midLeft: midLeftContact, midRight: midRightContact,
        backLeft: backLeftContact, backRight: backRightContact,
        stable: (frontLeftContact + frontRightContact + midLeftContact + midRightContact + backLeftContact + backRightContact) >= 4
      }
    }
    
    // Pad or truncate to exact sensor count
    while (sensorData.length < this.robotType.sensorCount) {
      sensorData.push(0)
    }
    sensorData = sensorData.slice(0, this.robotType.sensorCount)
    
    // Add ground contact info to sensor data object
    ;(sensorData as any).groundContact = groundContact
    
    return sensorData
  }

  // Comprehensive sensor data collection for different robot types
  getSensorData(bossRefs: any, deltaTime: number = 0.016): any {
    if (!bossRefs.head?.current || !bossRefs.torso?.current) return null
    
    const currentTime = performance.now()
    
    // Position sensors - adapt based on robot type
    const headPos = bossRefs.head.current.translation()
    const torsoPos = bossRefs.torso.current.translation()
    
    // Common limb positions for all robot types
    const limbPositions = this.getLimbPositions(bossRefs)
    
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

    // Create adaptive sensor data based on robot type
    const sensorData = this.createSensorData(headPos, torsoPos, limbPositions, headVel, torsoVel, headAccel, torsoAccel, torsoRot, torsoAngVel, bossRefs)

    // Store current state for next frame's acceleration calculation
    this.previousState = {
      headVel: { x: headVel.x, y: headVel.y, z: headVel.z },
      torsoVel: { x: torsoVel.x, y: torsoVel.y, z: torsoVel.z },
      timestamp: currentTime
    } as PreviousState

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

  // Generate enhanced motor control actions with exploration and smoothing
  async predict(sensorData: number[]): Promise<number[]> {
    this.stepCount++
    
    if (!this.model || !sensorData || sensorData.length < this.robotType.sensorCount) {
      // Return gentle random actions for exploration
      const action = this.generateExplorationAction()
      this.lastAction = [...action]
      return action
    }
    
    // Decay exploration rate gradually to let neural network take over
    this.explorationRate = Math.max(0.1, this.explorationRate * 0.999)
    
    let rawAction: number[]
    
    if (Math.random() < this.explorationRate) {
      // Exploration: try random actions
      rawAction = this.generateExplorationAction()
    } else {
      // Exploitation: use neural network prediction
      const stateTensor = tf.tensor2d([sensorData.slice(0, this.robotType.sensorCount)])
      const prediction = this.model.predict(stateTensor) as tf.Tensor
      const result = await prediction.data()
      
      stateTensor.dispose()
      prediction.dispose()
      
      rawAction = Array.from(result)
    }
    
    // Apply motor speed limiting to prevent rapid shaking
    const speedLimitedAction = rawAction.map((newVal, i) => {
      const currentVal = this.lastAction[i]
      const maxChange = this.maxMotorChangeRate
      const requestedChange = newVal - currentVal
      
      // Clamp the change to maximum allowed rate
      const limitedChange = Math.max(-maxChange, Math.min(maxChange, requestedChange))
      return currentVal + limitedChange
    })
    
    // Apply lighter smoothing on top of speed limiting
    const smoothingFactor = 0.3
    const smoothedAction = speedLimitedAction.map((newVal, i) => 
      this.lastAction[i] * smoothingFactor + newVal * (1 - smoothingFactor)
    )
    
    this.lastAction = [...smoothedAction]
    return smoothedAction
  }
  
  // Generate pure random exploration actions - let the neural network learn everything
  private generateExplorationAction(): number[] {
    const action = []
    for (let i = 0; i < this.robotType.motorCount; i++) {
      // Pure random values - no hardcoded patterns
      action.push((Math.random() - 0.5) * 2.0) // Range: -1.0 to 1.0 (full motor range)
    }
    return action
  }

  // Add training sample with fitness tracking (only during active training)
  addTrainingSample(state: number[], action: number[], fitness: number): void {
    if (!this.trainingActive) return // Only collect samples during active training
    
    this.trainingData.push({ state, action, fitness })
    this.fitnessHistory.push(fitness)
    
    // Track best performance
    if (fitness > this.bestFitness) {
      this.bestFitness = fitness
      this.bestAction = [...action]
      console.log(`New best episode fitness: ${fitness.toFixed(2)}`)
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

  // Start continuous training mode
  startTraining(): void {
    if (!this.isInitialized) {
      this.createModel()
    }
    this.trainingActive = true
    console.log('🎯 Training started - Boss will learn continuously')
  }
  
  // Pause continuous training mode
  pauseTraining(): void {
    this.trainingActive = false
    console.log('⏸️ Training paused')
  }
  
  // Train the model using collected data with fitness-weighted approach
  async trainModel(): Promise<void> {
    if (!this.model || this.trainingData.length < 5) return

    this.isTraining = true
    console.log('Training Boss AI with', this.trainingData.length, 'episode samples (not frame samples)...')
    
    // Filter and weight samples by fitness
    const sortedSamples = this.trainingData
      .filter(sample => sample.fitness > 20) // Only train on decent performance
      .sort((a, b) => b.fitness - a.fitness) // Best first
      .slice(0, 200) // Take best 200 samples
    
    if (sortedSamples.length < 3) {
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