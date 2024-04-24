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
    private var _frameOriginPose: Matrix4x4?
    private var _translationToOriginalFrame: Vector3 = .zero
    private var _positionLastTransmitted: Vector3 = .zero

    /// Set `true` to transmit poses to robot.
    @Published var transmitting: Bool = false {
        didSet {
            /*
             * We need to adjust for jumps in pose when transmission is disabled and then resumed.
             * Ultimately, we transmit the delta from the home pose (where tracking first started).
             * We accumulate an offset that erases the jumps. Visualized in 1D:
             *
             * +-----.       .------------------.       .--------------------+
             * home  stop1   start2             stop2   start3               current
             *
             *
             * offset = (current-start3) + (stop2-start2) + (stop1-home)
             *        = (current-home) + (stop1-start2) + (stop2-start3)
             *        = (current-home) + translationToHomeFrame
             *
             * Each time transmission is stopped and resumed, we accumulate:
             * stoppedPos - resumedPos
             */
            let currentPosition = _lastPose.position

            if transmitting {
                if _frameOriginPose == nil {
                    // First press: reset frame origin pose
                    _frameOriginPose = _lastPose
                    _translationToOriginalFrame = .zero
                } else {
                    // Subsequent press: we have jumped, and need to adjust offset back to original
                    // frame
                    _translationToOriginalFrame += _positionLastTransmitted - currentPosition
                }
            } else {
                _positionLastTransmitted = currentPosition
            }
        }
    }

    var translationScale: Float = 1.0 {
        didSet {
            log("Set translation scale: \(translationScale)")
        }
    }

    var gripperOpen: Float = 0.0 {
        didSet {
            _connection?.send(GripperOpenMessage(openAmount: gripperOpen))
        }
    }

    var gripperRotation: Float = 0.0 {
        didSet {
            _connection?.send(GripperRotateMessage(degrees: gripperRotation))
        }
    }

    init() {
        _task = Task {
            await runTask()
        }
    }

    func resetToHomePose() {
        transmitting = false
        _frameOriginPose = _lastPose
        _translationToOriginalFrame = .zero
        if let frameOriginPose = _frameOriginPose {
            _connection?.send(PoseUpdateMessage(initialPose: frameOriginPose, pose: _lastPose, deltaPosition: .zero))
            _connection?.send(GripperOpenMessage(openAmount: 0))
            _connection?.send(GripperRotateMessage(degrees: 0))
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
        if transmitting,
           let frameOriginPose = _frameOriginPose {
            let deltaPosition = (pose.position - frameOriginPose.position + _translationToOriginalFrame) / translationScale
            _connection?.send(PoseUpdateMessage(initialPose: frameOriginPose, pose: pose, deltaPosition: deltaPosition))
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
