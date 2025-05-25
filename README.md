# Boss AI Trainer 🤖

A cutting-edge browser-based robotics simulation platform for training neural networks to control humanoid robots. Built with Next.js, React Three Fiber, TensorFlow.js, and advanced physics simulation.

## 🌟 Features

### 🧠 Advanced AI Training
- **Real-time Neural Network Training** using TensorFlow.js in the browser
- **24 Comprehensive Sensors**: Position, velocity, acceleration, rotation, joint angles, ground contact
- **8 Motor Controls**: Torso forces, leg torques, balance adjustments
- **Fitness-based Learning**: AI learns to maximize head height and stability
- **20Hz Control Loop**: High-frequency sensor feedback for responsive control

### 🤖 Multi-Robot Architecture
- **Biped Boss**: Two-legged humanoid robot (24 sensors, 8 motors)
- **Quad Boss**: Four-legged robot *(Coming Soon - 32 sensors, 12 motors)*
- **Spider Boss**: Six-legged robot *(Coming Soon - 40 sensors, 18 motors)*
- **Extensible Design**: Easy to add new robot types

### 📦 Model Export & Sharing
- **Safetensors Export**: Download trained models in industry-standard format
- **Comprehensive Metadata**: Includes robot type, training stats, timestamps
- **Model Sharing**: Export and share trained AI controllers
- **Browser-based**: No server required for training or export

### 🎮 Interactive Simulation
- **Real-time Physics**: Powered by Rapier physics engine
- **WebGL Rendering**: Smooth 3D graphics with Three.js
- **Touch/Mouse Controls**: Orbit camera, zoom, pan on desktop and mobile
- **Computer Vision**: 128x128 WebGL frame capture for vision-based training

### 📊 Advanced Sensors & Analytics
- **IMU Simulation**: Acceleration, angular velocity, orientation data
- **Joint Encoders**: Real-time joint angle measurements
- **Ground Contact**: Foot pressure sensing
- **Balance Metrics**: Center of mass tracking
- **Real-time Visualization**: Live sensor data display with basic/detailed views

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and **Bun** package manager
- Modern browser with WebGL support
- **Recommended**: GPU acceleration for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd boss-ai-trainer
   ```

2. **Install dependencies**
   ```bash
   bun install
   ```

3. **Start development server**
   ```bash
   bun dev
   ```

4. **Open in browser**
   ```
   http://localhost:3000
   ```

## 🎯 How to Use

### 1. Basic Training Workflow

1. **Launch the app** and wait for TensorFlow.js to initialize
2. **Select robot type** from the dropdown (currently only Biped Boss available)
3. **Watch Boss fall** and collect sensor data automatically
4. **Wait for 50+ samples** to accumulate for training
5. **Click "Train AI"** to improve Boss's balance
6. **Repeat training** as Boss learns to stay upright
7. **Export your model** when satisfied with performance

### 2. Sensor Data Monitoring

**Basic View:**
- Head height, torso position, center of mass, ground contact status

**Detailed View:**
- **Position Sensors**: All body part coordinates
- **Velocity Sensors**: Movement vectors for head and torso
- **Acceleration**: Dynamic motion analysis
- **Rotation**: Orientation quaternions and angular velocity
- **Joint Angles**: Knee positions in degrees
- **Ground Contact**: Individual foot contact status

### 3. Model Export & Sharing

1. **Train your model** with at least 50 samples (more = better)
2. **Click "📦 Export Model"** button
3. **Download** the `.safetensors.json` file automatically
4. **Share** your trained model with others
5. **Import models** *(feature coming soon)*

### 4. Camera Controls

**Desktop:**
- **Mouse drag**: Orbit camera around Boss
- **Scroll wheel**: Zoom in/out
- **Right-click drag**: Pan camera

**Mobile/Touch:**
- **Touch drag**: Orbit camera
- **Pinch**: Zoom in/out
- **Two-finger drag**: Pan camera

## 🧠 AI Architecture

### Neural Network Structure
```
Input Layer:     24 sensors (robot-dependent)
Hidden Layer 1:  64 neurons (ReLU) + 20% Dropout
Hidden Layer 2:  32 neurons (ReLU) + 20% Dropout  
Hidden Layer 3:  16 neurons (ReLU)
Output Layer:    8 motors (Tanh activation)
```

### Sensor Input Vector (24D for Biped)
```javascript
[
  // Position sensors (6)
  head_y, torso_y, left_thigh_y, right_thigh_y, center_mass_y, torso_x,
  
  // Velocity sensors (6) 
  head_vel_x, head_vel_y, head_vel_z, torso_vel_x, torso_vel_y, torso_vel_z,
  
  // Acceleration sensors (4)
  head_accel_y, torso_accel_x, torso_accel_y, torso_accel_z,
  
  // Rotation sensors (4)
  torso_rot_x, torso_rot_y, torso_rot_z, torso_rot_w,
  
  // Angular velocity (2)
  torso_ang_vel_x, torso_ang_vel_z,
  
  // Joint angles (2)
  left_knee_angle, right_knee_angle
]
```

### Motor Output Vector (8D for Biped)
```javascript
[
  torso_torque_x,     // Main balance control
  torso_torque_y,     // Upward force  
  torso_torque_z,     // Forward/backward balance
  force_strength,     // Force multiplier
  left_leg_torque,    // Left leg motor
  right_leg_torque,   // Right leg motor
  balance_adjust_x,   // Fine balance tuning
  balance_adjust_z    // Stability control
]
```

### Fitness Function
```javascript
fitness = upright_bonus +           // Head height above 1.0
          torso_height_bonus +      // Torso stability
          center_mass_stability +   // Lateral balance
          velocity_penalty +        // Penalize excessive motion
          acceleration_penalty +    // Smooth movement reward
          orientation_bonus +       // Upright orientation
          joint_efficiency +        // Natural joint angles
          ground_contact_bonus      // Both feet on ground
```

## 🛠️ Technical Stack

### Core Technologies
- **Next.js 15** - React framework with App Router
- **React 19** - Latest React with concurrent features
- **TensorFlow.js 4.22** - Browser-based machine learning
- **React Three Fiber** - Three.js React renderer
- **React Three Drei** - Three.js utilities and helpers
- **React Three Rapier** - Physics simulation
- **Tailwind CSS** - Utility-first CSS framework
- **Bun** - Fast JavaScript runtime and package manager

### AI & ML Libraries
- **@tensorflow/tfjs** - Neural network training and inference
- **@react-three/rapier** - Physics-based sensor simulation

### 3D Graphics & Physics
- **Three.js** - WebGL 3D graphics
- **Rapier** - Rust-based physics engine (WASM)
- **WebGL** - GPU-accelerated rendering

## 📁 Project Structure

```
boss-ai-trainer/
├── src/
│   ├── app/
│   │   ├── page.tsx             # Main application page
│   │   ├── layout.tsx           # App layout
│   │   └── globals.css          # Global styles
│   ├── components/
│   │   ├── Boss.js              # Main Boss component
│   │   ├── BossController.js    # Neural network controller
│   │   ├── AIBoss.js           # Physics robot component
│   │   ├── AIControlPanel.js   # Training interface
│   │   ├── Scene.js            # 3D scene setup
│   │   ├── CanvasCapture.js    # Computer vision capture
│   │   └── ui/                 # UI components
│   └── lib/
│       ├── ModelExporter.js    # Safetensors export utility
│       ├── robotTypes.js       # Robot configurations
│       └── utils.ts            # Utility functions
├── public/                     # Static assets
├── package.json
└── README.md
```

## 🎛️ Configuration

### Robot Types
Add new robot types in `ROBOT_TYPES` configuration:

```javascript
const ROBOT_TYPES = {
  YOUR_ROBOT: {
    id: 'your_robot',
    name: 'Your Robot Name',
    description: 'Robot description',
    sensorCount: 24,    // Number of sensor inputs
    motorCount: 8,      // Number of motor outputs  
    modelSize: 'small'  // Model complexity
  }
}
```

### Training Parameters
Adjust in `BossController`:

```javascript
// Neural network architecture
const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [sensorCount], units: 64 }),
    // ... customize layers
  ]
})

// Training hyperparameters
model.compile({
  optimizer: tf.train.adam(0.0005),  // Learning rate
  loss: 'meanSquaredError',
  metrics: ['mse']
})
```

## 🔧 Development

### Adding New Robot Types

1. **Define robot configuration** in `ROBOT_TYPES`
2. **Create robot component** with appropriate sensors/motors
3. **Implement sensor collection** for new robot morphology
4. **Add motor control logic** for new actuators
5. **Update fitness function** for robot-specific goals

### Extending Sensor Systems

```javascript
// Add new sensor types
getSensorData(bossRefs, deltaTime) {
  const sensors = [
    ...existingSensors,
    newSensorType1,        // Add your sensor
    newSensorType2,        // Add another sensor
  ]
  return sensors
}
```

### Custom Fitness Functions

```javascript
// Modify fitness calculation
calculateFitness(sensorData) {
  const customObjective = yourCustomLogic(sensorData)
  return baseScore + customObjective
}
```

## 📊 Performance Tips

### Training Optimization
- **Collect 100+ samples** before training for better results
- **Train multiple times** as data accumulates
- **Monitor fitness score** to track learning progress
- **Reset if Boss gets stuck** in local minima

### Browser Performance  
- **Use Chrome/Edge** for best TensorFlow.js performance
- **Close other tabs** to free GPU memory
- **Enable hardware acceleration** in browser settings
- **Use dedicated GPU** if available

### Model Size vs Performance
- **Small models** (current): Fast training, good for real-time
- **Medium models**: Better performance, slower training
- **Large models**: Best results, requires more compute

## 🐛 Troubleshooting

### Common Issues

**Model not training:**
- Check if 50+ samples collected
- Verify TensorFlow.js backend loaded
- Try refreshing page to reset state

**Poor AI performance:**
- Collect more training data (200+ samples)
- Train multiple times
- Adjust fitness function weights
- Reset and start fresh training

**Export fails:**
- Ensure model is trained (green status)
- Check browser download permissions
- Try with fewer browser tabs open

**Physics glitches:**
- Reset simulation with "Reset Boss" button
- Check for browser WebGL issues
- Ensure stable framerate (close other apps)

### Browser Compatibility
- **✅ Chrome 90+** (Recommended)
- **✅ Firefox 88+** 
- **✅ Safari 14+**
- **✅ Edge 90+**
- **❌ Internet Explorer** (Not supported)

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Code Style
- **ES6+ JavaScript** with modern syntax
- **React Hooks** over class components
- **Functional programming** where possible
- **TypeScript** welcome for new features
- **ESLint** configuration provided

### Testing
- **Manual testing** in multiple browsers
- **Performance testing** with various hardware
- **AI training validation** across different scenarios

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TensorFlow.js Team** - Browser-based machine learning
- **Three.js Community** - Amazing 3D graphics library  
- **Rapier Physics** - High-performance physics simulation
- **React Three Fiber** - Declarative 3D in React
- **Robotics Community** - Inspiration for sensor systems

## 🔗 Related Projects

- **[TensorFlow.js Examples](https://github.com/tensorflow/tfjs-examples)** - More ML examples
- **[React Three Fiber](https://github.com/pmndrs/react-three-fiber)** - 3D React renderer
- **[Rapier Physics](https://rapier.rs/)** - Physics engine documentation
- **[Three.js](https://threejs.org/)** - 3D graphics library

## 📞 Support

- **GitHub Issues** - Bug reports and feature requests
- **Discussions** - Community support and questions
- **Documentation** - In-code comments and examples

---

**Happy Robot Training!** 🤖✨

*Train your AI, export your models, and push the boundaries of browser-based robotics simulation.*