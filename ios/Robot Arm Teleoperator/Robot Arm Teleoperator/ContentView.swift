//
//  ContentView.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

import Combine
import SwiftUI
import RealityKit

struct ContentView : View {
    @StateObject var teleoperation = Teleoperation()

    @State private var _translationScale = 1.2
    @State private var _initialGripperRotationValue = NonReactiveState<Float>(wrappedValue: 0)
    @State private var _initialGripperOpenValue = NonReactiveState<Float>(wrappedValue: 0)

    var body: some View {
        ZStack {
            VStack {
                ARViewContainer(teleoperation: teleoperation)
                    .edgesIgnoringSafeArea(.all)
            }

            GeometryReader { geometry in
                VStack {
                    // Settings atop display
                    HStack {
                        // Two columns
                        VStack {
                            Text("Translation Scale")
                            HStack {
                                Slider(
                                    value: $_translationScale,
                                    in: 1...2
                                )
                                .onChange(of: _translationScale, initial: true) {
                                    teleoperation.translationScale = Float(_translationScale)
                                }
                                Text("\(String(format: "%1.1f", _translationScale))")
                            }
                            Spacer()
                        }
                        .frame(width: geometry.size.width / 2) // set slider width to half of display width

                        VStack {
                            Button(action: {
                                teleoperation.resetToHomePose()
                            }) {
                                Label("Home Pose", systemImage: "house.circle")
                                    .font(.title)
                            }
                            Spacer()
                        }
                    }

                    // Gripper control regions
                    ZStack {
                        Rectangle()
                            .fill(.purple.opacity(0.5))
                            .onHorizontalDrag { started, startPosition, currentPosition in
                                // Incrementally rotate the gripper based on graction of view width
                                // swiped
                                if started {
                                    _initialGripperRotationValue.wrappedValue = teleoperation.gripperRotation
                                } else {
                                    let delta = 90 * (currentPosition - startPosition)
                                    teleoperation.gripperRotation = clamp(_initialGripperRotationValue.wrappedValue + delta, min: -90, max: 90)
                                }
                            }
                        Text("Gripper Rotation")
                            .foregroundStyle(.purple)
                    }

                    ZStack {
                        Rectangle()
                            .fill(.blue.opacity(0.5))
                            .onHorizontalDrag { started, startPosition, currentPosition in
                                // Incrementally move the gripper based on fraction of view width
                                // swiped
                                if started {
                                    _initialGripperOpenValue.wrappedValue = teleoperation.gripperOpen
                                } else {
                                    let delta = currentPosition - startPosition
                                    teleoperation.gripperOpen = clamp(_initialGripperOpenValue.wrappedValue + delta, min: 0, max: 1)
                                }
                            }
                        Text("Gripper Open")
                            .foregroundStyle(.red)
                    }

                    ZStack {
                        Rectangle()
                            .fill((teleoperation.moving ? Color.red : Color.green).opacity(0.5))
                            .onTouchDown {
                                teleoperation.moving = !teleoperation.moving
                            }
                        Text(teleoperation.moving ? "Stop" : "Move")
                    }
                }
            }
        }
    }
}

struct ARViewContainer: UIViewRepresentable {
    @ObservedObject private var _teleoperation: Teleoperation

    init(teleoperation: Teleoperation) {
        _teleoperation = teleoperation
    }

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)

        // Create a cube model
        let mesh = MeshResource.generateBox(size: 0.1, cornerRadius: 0.005)
        let material = SimpleMaterial(color: .gray, roughness: 0.15, isMetallic: true)
        let model = ModelEntity(mesh: mesh, materials: [material])
        model.transform.translation.y = 0.05

        // Create horizontal plane anchor for the content
        let anchor = AnchorEntity(.plane(.horizontal, classification: .any, minimumBounds: SIMD2<Float>(0.2, 0.2)))
        anchor.children.append(model)

        // Add the horizontal plane anchor to the scene
        arView.scene.anchors.append(anchor)

        // Subscribe events
        _teleoperation.subscribeToEvents(from: arView)

        return arView
    }
    
    func updateUIView(_ uiView: ARView, context: Context) {
    }
}

#Preview {
    ContentView()
}
