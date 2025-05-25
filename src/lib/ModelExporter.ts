// Safetensors export utility
import * as tf from '@tensorflow/tfjs';

interface TensorData {
  dtype: string;
  shape: number[];
  data: Float32Array;
}

interface ExportMetadata {
  format?: string;
  framework?: string;
  created_at?: string;
  model_type?: string;
  architecture?: string;
  [key: string]: any;
}

interface ExportData {
  metadata: ExportMetadata;
  tensors: Record<string, TensorData>;
  version: string;
}

export class ModelExporter {
  static async exportToSafetensors(model: tf.LayersModel, metadata: ExportMetadata = {}): Promise<Blob> {
    if (!model) {
      throw new Error('No model to export')
    }

    try {
      // Get model weights
      const weights = model.getWeights()
      const tensors: Record<string, TensorData> = {}

      // Convert TensorFlow.js tensors to Safetensors format
      for (let i = 0; i < weights.length; i++) {
        const weight = weights[i]
        const name = `layer_${i}_${(weight as any).name || 'weight'}`
        const data = await weight.data()
        
        // Convert to Float32Array if needed
        const float32Data = data instanceof Float32Array ? data : new Float32Array(data)
        
        tensors[name] = {
          dtype: 'F32',
          shape: weight.shape,
          data: float32Data
        }
      }

      // Create metadata
      const fullMetadata: ExportMetadata = {
        format: 'tf.js',
        framework: 'tensorflow.js',
        created_at: new Date().toISOString(),
        model_type: 'sequential',
        architecture: 'robotics_controller',
        ...metadata
      }

      // Create a simple safetensors-like format (simplified implementation)
      const exportData: ExportData = {
        metadata: fullMetadata,
        tensors: tensors,
        version: '1.0'
      }

      // Convert to JSON string for download
      const jsonString = JSON.stringify(exportData, null, 2)
      const blob = new Blob([jsonString], { type: 'application/json' })
      
      // Dispose of weights
      weights.forEach(w => w.dispose())
      
      return blob

    } catch (error) {
      console.error('Export failed:', error)
      throw new Error(`Failed to export model: ${(error as Error).message}`)
    }
  }

  static downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = filename
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }
}