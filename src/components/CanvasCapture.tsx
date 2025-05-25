import { useRef, useEffect } from 'react'
import { useThree, useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface CanvasCaptureProps {
  onCapture: (pixels: Uint8Array) => void;
  isCapturing: boolean;
}

// Canvas capture component for computer vision
export function CanvasCapture({ onCapture, isCapturing }: CanvasCaptureProps) {
  const { gl, scene, camera } = useThree()
  const renderTarget = useRef<THREE.WebGLRenderTarget | null>(null)

  useEffect(() => {
    // Create render target for capturing frames
    renderTarget.current = new THREE.WebGLRenderTarget(128, 128)
  }, [])

  useFrame(() => {
    if (!isCapturing || !renderTarget.current) return

    // Render scene to texture
    const originalRenderTarget = gl.getRenderTarget()
    gl.setRenderTarget(renderTarget.current)
    gl.render(scene, camera)
    gl.setRenderTarget(originalRenderTarget)

    // Capture image data
    const pixels = new Uint8Array(128 * 128 * 4)
    gl.readRenderTargetPixels(renderTarget.current, 0, 0, 128, 128, pixels)
    
    onCapture(pixels)
  })

  return null
}