'use client'

import { Canvas } from '@react-three/fiber'
import { useState, useEffect, useCallback } from 'react'
import * as tf from '@tensorflow/tfjs'
import { BossController } from './BossController'
import { AIControlPanel } from './AIControlPanel'
import { Scene } from './Scene'
import { ROBOT_TYPES } from '@/lib/robotTypes'

interface SensorData {
  sensors: number[];
  groundContact?: {
    left: number;
    right: number;
    both: boolean;
  };
  fitness?: number;
}

export default function Boss() {
  const [isClient, setIsClient] = useState(false)
  const [resetKey, setResetKey] = useState(0)
  const [controller] = useState(() => new BossController())
  const [currentState, setCurrentState] = useState<number[] | null>(null)
  const [sensorData, setSensorData] = useState<SensorData | null>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [captureCount, setCaptureCount] = useState(0)
  const [selectedRobotType, setSelectedRobotType] = useState('biped')

  useEffect(() => {
    setIsClient(true)
    // Initialize TensorFlow.js
    tf.ready().then(() => {
      console.log('TensorFlow.js ready!')
      console.log('Backend:', tf.getBackend())
      console.log('Enhanced sensor system: 24 inputs, 8 motor outputs')
    })

    // Listen for boundary exit events
    const handleBoundaryExit = () => {
      console.log('Ragdoll exited boundaries - resetting...')
      handleReset()
    }

    window.addEventListener('ragdollBoundaryExit', handleBoundaryExit)
    
    return () => {
      window.removeEventListener('ragdollBoundaryExit', handleBoundaryExit)
    }
  }, [])

  const handleReset = () => {
    setResetKey(prev => prev + 1)
    controller.resetPositionState() // Only reset position, keep training data
  }

  const handleModelReset = () => {
    controller.resetModel()
    console.log('Neural network model reset - training data preserved')
  }

  const handleCompleteReset = () => {
    setResetKey(prev => prev + 1)
    controller.resetAll()
    console.log('Complete reset - starting fresh')
  }

  const handleSensorUpdate = useCallback((newSensorData: SensorData) => {
    setSensorData(newSensorData)
  }, [])

  const handleCapture = useCallback((pixels: Uint8Array) => {
    // Process captured frame (could be used for vision-based training)
    setCaptureCount(prev => prev + 1)
    // Here you could process the pixels for vision-based AI training
    console.log(`Captured frame ${captureCount}: ${pixels.length} pixels`)
  }, [captureCount])

  const handleToggleCapture = () => {
    setIsCapturing(prev => !prev)
  }

  const handleRobotTypeChange = (robotTypeId: string) => {
    setSelectedRobotType(robotTypeId)
    // Note: This would require recreating the controller with new robot type
    // For now, we'll just update the selection (actual robot switching not implemented)
    console.log('Robot type changed to:', robotTypeId)
  }

  if (!isClient) {
    return (
      <div className="w-full h-screen flex items-center justify-center bg-gray-900 text-white">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto mb-4"></div>
          <p>Loading Boss...</p>
        </div>
      </div>
    )
  }

  const currentRobotType = ROBOT_TYPES[selectedRobotType.toUpperCase()] || ROBOT_TYPES.BIPED

  return (
    <div className="w-full h-screen bg-gray-900 overflow-hidden">
      {/* Reset Controls */}
      <div className="absolute top-16 right-4 z-20 flex flex-col gap-2">
        <button
          onClick={handleReset}
          className="bg-orange-600 hover:bg-orange-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 shadow-lg"
        >
          🔄 Reset Position
        </button>
        <button
          onClick={handleModelReset}
          className="bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 shadow-lg"
        >
          🧠 Reset Model
        </button>
        <button
          onClick={handleCompleteReset}
          className="bg-red-600 hover:bg-red-700 text-white px-3 py-2 rounded-lg text-sm font-medium transition-colors duration-200 shadow-lg"
        >
          💥 Reset All
        </button>
      </div>

      {/* Enhanced AI Control Panel */}
      <AIControlPanel 
        controller={controller} 
        currentState={currentState}
        sensorData={sensorData}
        onToggleCapture={handleToggleCapture}
        isCapturing={isCapturing}
        onRobotTypeChange={handleRobotTypeChange}
        selectedRobotType={selectedRobotType}
      />

      {/* 3D Canvas */}
      <Canvas
        camera={{ 
          position: [5, 5, 5], 
          fov: 60,
          near: 0.1,
          far: 1000
        }}
        gl={{ 
          antialias: true,
          preserveDrawingBuffer: true, // Required for capturing
          powerPreference: "high-performance"
        }}
        shadows={true}
        className="w-full h-full"
      >
        <Scene 
          resetKey={resetKey} 
          controller={controller}
          onStateUpdate={setCurrentState}
          onSensorUpdate={handleSensorUpdate}
          onCapture={handleCapture}
          isCapturing={isCapturing}
        />
      </Canvas>
    </div>
  )
}