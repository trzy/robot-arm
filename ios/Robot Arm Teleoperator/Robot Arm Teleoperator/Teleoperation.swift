//
//  Teleoperation.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

import Combine
import RealityKit
import SwiftUI

class Teleoperation: ObservableObject {
    private var _task: Task<Void, Never>!
    private var _subscriptions: [Cancellable] = []
    private var _connection: AsyncTCPConnection?
    private var _lastPose: Matrix4x4 = .identity
    private var _initialPose: Matrix4x4 = .identity

    /// Set `true` to transmit poses to robot. Each time this is changed from `false` to `true`,
    /// the reference pose is reset to the current pose. Pose updates will be relative to it.
    var transmitting: Bool = false {
        didSet {
            _initialPose = _lastPose
        }
    }

    var translationScale: Float = 1.0 {
        didSet {
            log("Set translation scale: \(translationScale)")
        }
    }

    init() {
        _task = Task {
            await runTask()
        }
    }

    func subscribeToEvents(from view: ARView) {
        // Subscribe to frame updates
        _subscriptions.append(view.scene.subscribe(to: SceneEvents.Update.self) { [weak self] event in
            let pose: Matrix4x4 = view.cameraTransform.matrix
            self?.onUpdate(event: event, pose: pose)
        })
    }

    func onUpdate(event: SceneEvents.Update, pose: Matrix4x4) {
        _lastPose = pose
        if transmitting {
            let deltaPosition = (pose.position - _initialPose.position) / translationScale
            _connection?.send(PoseUpdateMessage(initialPose: _initialPose, pose: pose, deltaPosition: deltaPosition))
        }
    }

    private func runTask() async {
        while true {
            do {
                let connection = try await AsyncTCPConnection(host: "10.104.162.243", port: 8000)
                _connection = connection
                connection.send(HelloMessage(message: "Hello from iOS!"))
                for try await receivedMessage in connection {
                    await handleMessage(receivedMessage, connection: connection)
                }
            } catch {
                log("Error: \(error.localizedDescription)")
            }
            _connection = nil
            try? await Task.sleep(for: .seconds(5))
        }
    }

    private func handleMessage(_ receivedMessage: ReceivedJSONMessage, connection: AsyncTCPConnection) async {
        switch receivedMessage.id {
        case HelloMessage.id:
            if let msg = JSONMessageDeserializer.decode(receivedMessage, as: HelloMessage.self) {
                log("Hello received: \(msg.message)")
            }

        default:
            log("Error: Unhandled message: \(receivedMessage.id)")
            break
        }
    }
}

fileprivate func log(_ message: String) {
    print("[Teleoperation] \(message)")
}
