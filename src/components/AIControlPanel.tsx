import { useState, useEffect } from 'react'
import { ROBOT_TYPES } from '@/lib/robotTypes'
import { BossController } from './BossController'

interface SensorData {
  sensors: number[];
  groundContact?: {
    left: number;
    right: number;
    both: boolean;
  };
  fitness?: number;
}

interface AIControlPanelProps {
  controller: BossController;
  currentState?: number[] | null;
  sensorData?: SensorData | null;
  onToggleCapture: () => void;
  isCapturing: boolean;
  onRobotTypeChange?: (robotTypeId: string) => void;
  selectedRobotType: string;
}

// Enhanced AI Control Panel with model export and robot selection
export function AIControlPanel({ controller, currentState: _currentState, sensorData, onToggleCapture, isCapturing, onRobotTypeChange, selectedRobotType }: AIControlPanelProps) {
  const [isModelReady, setIsModelReady] = useState(false)
  const [trainingStats, setTrainingStats] = useState({ samples: 0, fitness: 0 })
  const [sensorDisplay, setSensorDisplay] = useState('basic')
  const [exportStatus, setExportStatus] = useState('')

  useEffect(() => {
    if (!controller.isInitialized) {
      controller.createModel()
      setIsModelReady(true)
    }
  }, [controller])

  useEffect(() => {
    if (sensorData) {
      setTrainingStats({
        samples: controller.trainingData.length,
        fitness: parseFloat(sensorData.fitness?.toFixed(2) || '0.00')
      })
    }
  }, [sensorData, controller])

  const handleTrain = async () => {
    await controller.trainModel()
  }

  const handleExportModel = async () => {
    setExportStatus('Exporting...')
    try {
      const result = await controller.exportModel()
      if (result.success) {
        setExportStatus(`✅ Exported: ${result.filename}`)
        setTimeout(() => setExportStatus(''), 3000)
      } else {
        setExportStatus(`❌ Error: ${result.error}`)
        setTimeout(() => setExportStatus(''), 5000)
      }
    } catch (error) {
      setExportStatus(`❌ Export failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      setTimeout(() => setExportStatus(''), 5000)
    }
  }

  const toggleSensorDisplay = () => {
    setSensorDisplay(prev => prev === 'basic' ? 'detailed' : 'basic')
  }

  return (
    <div className="absolute top-20 left-4 bg-black bg-opacity-90 text-white p-4 rounded-lg text-xs max-w-sm space-y-3 max-h-[75vh] overflow-y-auto z-50">
      <h3 className="font-bold text-green-400 text-sm">🤖 Boss AI Control</h3>
      
      {/* Robot Type Selection */}
      {onRobotTypeChange && (
        <div className="space-y-2">
          <label className="text-blue-300 text-xs">🔧 Robot Type:</label>
          <select
            value={selectedRobotType}
            onChange={(e) => onRobotTypeChange(e.target.value)}
            className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-xs text-white"
          >
            {Object.values(ROBOT_TYPES).map(robotType => (
              <option 
                key={robotType.id} 
                value={robotType.id}
                disabled={robotType.disabled}
              >
                {robotType.name} - {robotType.description}
                {robotType.disabled ? ' (Coming Soon)' : ''}
              </option>
            ))}
          </select>
          <div className="text-xs text-gray-400">
            Sensors: {controller.robotType.sensorCount} | Motors: {controller.robotType.motorCount}
          </div>
        </div>
      )}
      
      <div className="space-y-1">
        <div>Status: {isModelReady ? '✅ Ready' : '🔄 Loading...'}</div>
        <div>Training: {controller.isTraining ? '🎯 Active' : '⏸️ Idle'}</div>
        <div>Samples: {trainingStats.samples}</div>
        <div>Fitness: {trainingStats.fitness}</div>
        {trainingStats.samples > 0 && (
          <div className="text-green-300 text-xs">
            📚 Learning from {trainingStats.samples} attempts
          </div>
        )}
      </div>

      {sensorData && (
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <div className="text-blue-300 text-xs">📊 Sensor Data:</div>
            <button
              onClick={toggleSensorDisplay}
              className="text-xs bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
            >
              {sensorDisplay === 'basic' ? 'Detailed' : 'Basic'}
            </button>
          </div>

          {sensorDisplay === 'basic' ? (
            <div className="space-y-1">
              <div>Head: {sensorData.sensors[0]?.toFixed(2)}</div>
              <div>Torso: {sensorData.sensors[1]?.toFixed(2)}</div>
              <div>CoM: {sensorData.sensors[4]?.toFixed(2)}</div>
              <div>Ground: {sensorData.groundContact?.both ? '✅' : '❌'}</div>
            </div>
          ) : (
            <div className="space-y-1 text-xs">
              <div className="text-yellow-300">🏗️ Position Sensors:</div>
              <div>Head Y: {sensorData.sensors[0]?.toFixed(3)}</div>
              <div>Torso Y: {sensorData.sensors[1]?.toFixed(3)}</div>
              <div>L.Thigh: {sensorData.sensors[2]?.toFixed(3)}</div>
              <div>R.Thigh: {sensorData.sensors[3]?.toFixed(3)}</div>
              <div>Center Mass: {sensorData.sensors[4]?.toFixed(3)}</div>
              <div>Torso X: {sensorData.sensors[5]?.toFixed(3)}</div>
              
              <div className="text-cyan-300 mt-2">🚀 Velocity Sensors:</div>
              <div>Head Vel: {sensorData.sensors[6]?.toFixed(2)}, {sensorData.sensors[7]?.toFixed(2)}, {sensorData.sensors[8]?.toFixed(2)}</div>
              <div>Torso Vel: {sensorData.sensors[9]?.toFixed(2)}, {sensorData.sensors[10]?.toFixed(2)}, {sensorData.sensors[11]?.toFixed(2)}</div>
              
              <div className="text-red-300 mt-2">⚡ Acceleration:</div>
              <div>Head Acc Y: {sensorData.sensors[12]?.toFixed(2)}</div>
              <div>Torso Acc: {sensorData.sensors[13]?.toFixed(2)}, {sensorData.sensors[14]?.toFixed(2)}, {sensorData.sensors[15]?.toFixed(2)}</div>
              
              <div className="text-purple-300 mt-2">🔄 Rotation:</div>
              <div>Torso Rot: {sensorData.sensors[16]?.toFixed(2)}, {sensorData.sensors[17]?.toFixed(2)}, {sensorData.sensors[18]?.toFixed(2)}</div>
              <div>Ang Vel: {sensorData.sensors[20]?.toFixed(2)}, {sensorData.sensors[21]?.toFixed(2)}</div>
              
              <div className="text-green-300 mt-2">🦵 Joint Angles:</div>
              <div>L.Knee: {(sensorData.sensors[22] * 57.3)?.toFixed(1)}°</div>
              <div>R.Knee: {(sensorData.sensors[23] * 57.3)?.toFixed(1)}°</div>
              
              <div className="text-orange-300 mt-2">👣 Ground Contact:</div>
              <div>Left: {sensorData.groundContact?.left ? '✅' : '❌'} Right: {sensorData.groundContact?.right ? '✅' : '❌'}</div>
            </div>
          )}
        </div>
      )}

      <div className="flex flex-col gap-2">
        <button
          onClick={handleTrain}
          disabled={controller.isTraining || trainingStats.samples < 50}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 px-3 py-1 rounded text-xs font-medium transition-colors"
        >
          {controller.isTraining ? 'Training...' : `Train AI (${Math.max(0, 50 - trainingStats.samples)} more needed)`}
        </button>
        
        <button
          onClick={handleExportModel}
          disabled={!controller.isInitialized || controller.isTraining || trainingStats.samples < 10}
          className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 px-3 py-1 rounded text-xs font-medium transition-colors"
        >
          📦 Export Model (.safetensors)
        </button>
        
        <button
          onClick={onToggleCapture}
          className={`px-3 py-1 rounded text-xs font-medium transition-colors ${
            isCapturing 
              ? 'bg-red-600 hover:bg-red-700' 
              : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          📷 {isCapturing ? 'Stop Vision' : 'Start Vision'}
        </button>
      </div>

      {exportStatus && (
        <div className={`text-xs p-2 rounded ${
          exportStatus.includes('✅') ? 'bg-green-800' : 
          exportStatus.includes('❌') ? 'bg-red-800' : 
          'bg-blue-800'
        }`}>
          {exportStatus}
        </div>
      )}

      <div className="text-xs text-gray-300 space-y-1">
        <div>💡 Train → Export → Share your AI models!</div>
        <div className="border-t border-gray-600 pt-2 mt-2">
          <div className="text-yellow-300">🔄 Reset Types:</div>
          <div>🔄 Position: Robot only (keeps learning)</div>
          <div>🧠 Model: Neural net only (keeps data)</div>
          <div>💥 All: Fresh start</div>
        </div>
      </div>
    </div>
  )
}