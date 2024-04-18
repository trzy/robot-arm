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

    var body: some View {
        ARViewContainer(teleoperation: teleoperation)
            .edgesIgnoringSafeArea(.all)
            .onTouchDown {
                teleoperation.transmitting = true
            }
            .onTouchUp {
                teleoperation.transmitting = false
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
