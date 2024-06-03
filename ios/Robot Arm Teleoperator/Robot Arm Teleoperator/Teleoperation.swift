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
    private var _tasks: [Task<Void, Never>] = []
    private var _subscriptions: [Cancellable] = []
    private var _reliableConnection: (any AsyncConnection)?
    private var _unreliableConnection: (any AsyncConnection)?
    private var _poseLastTransmittedAt: TimeInterval = 0
    private var _lastPose: Matrix4x4 = .identity
    private var _lastTransmittedPose: Matrix4x4 = .identity
    private var _frameOriginPose: Matrix4x4?
    private var _translationToOriginalFrame: Vector3 = .zero
    private var _positionLastTransmitted: Vector3 = .zero

    /// Set `true` to transmit poses to robot.
    @Published var moving: Bool = false {
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

            if moving {
                if _frameOriginPose == nil {
                    // First press: reset frame origin pose
                    _frameOriginPose = _lastPose
                    _translationToOriginalFrame = .zero
                } else {
                    // Subsequent press: we have jumped, and need to adjust offset back to original
                    // frame
                    _translationToOriginalFrame += _positionLastTransmitted - currentPosition
                }

                if !oldValue {
                    // On every stopped -> moving transition, send start-of-episode signal
                    _reliableConnection?.send(BeginEpisodeMessage())
                }
            } else {
                _positionLastTransmitted = currentPosition
                if oldValue {
                    // On every moving -> stopped transition, send end-of-episode signal
                    _reliableConnection?.send(EndEpisodeMessage())
                }
            }
        }
    }

    var translationScale: Float = 1.0 {
        didSet {
            log("Set translation scale: \(translationScale)")
        }
    }

    var gripperOpen: Float = 0.0
    var gripperRotation: Float = 0.0

    init() {
        _tasks.append(Task {
            await runReliableConnectionTask()
        })
        _tasks.append(Task {
            await runUnreliableConnectionTask()
        })
    }

    func resetToHomePose() {
        moving = false
        _frameOriginPose = nil
        _translationToOriginalFrame = .zero
        _reliableConnection?.send(PoseStateMessage(gripperDeltaPosition: .zero, gripperOpenAmount: 0, gripperRotateDegrees: 0))
    }

    func subscribeToEvents(from view: ARView) {
        // Subscribe to frame updates
        _subscriptions.append(view.scene.subscribe(to: SceneEvents.Update.self) { [weak self] event in
            let pose: Matrix4x4 = view.cameraTransform.matrix
            self?.onUpdate(event: event, pose: pose)
        })
    }

    func onUpdate(event: SceneEvents.Update, pose: Matrix4x4) {
        // Throttle transmission rate. We are using TCP and both the IK solver and actual motor
        // servos take time to respond. No need to spam the connection.
        let transmissionHz = 20.0
        let period = 1.0 / transmissionHz
        let now = Date().timeIntervalSinceReferenceDate
        let timeSinceLastTransmission = now - _poseLastTransmittedAt
        guard timeSinceLastTransmission >= period else { return }

        // Transmit update
        _lastPose = pose
        let currentPose = moving ? pose : _lastTransmittedPose    // when not moving, keep transmitting old pose (and always update gripper)
        if let frameOriginPose = _frameOriginPose {
            let deltaPosition = (currentPose.position - frameOriginPose.position + _translationToOriginalFrame) / translationScale
            _unreliableConnection?.send(PoseStateMessage(gripperDeltaPosition: deltaPosition, gripperOpenAmount: gripperOpen, gripperRotateDegrees: gripperRotation))
            _lastTransmittedPose = currentPose
        } else {
            // We haven't moved yet, just transmit gripper
            _unreliableConnection?.send(PoseStateMessage(gripperDeltaPosition: .zero, gripperOpenAmount: gripperOpen, gripperRotateDegrees: gripperRotation))
        }
        _poseLastTransmittedAt = now
    }

    private func runReliableConnectionTask() async {
        while true {
            do {
                let connection = try await AsyncTCPConnection(host: Settings.shared.host, port: Settings.shared.port)
                _reliableConnection = connection
                connection.send(HelloMessage(message: "Hello from iOS over TCP!"))
                for try await receivedMessage in connection {
                    await handleMessage(receivedMessage, connection: connection)
                }
            } catch {
                log("Error: \(error.localizedDescription)")
            }
            _reliableConnection = nil
            try? await Task.sleep(for: .seconds(5))
        }
    }

    private func runUnreliableConnectionTask() async {
        while true {
            do {
                let connection = try await AsyncUDPConnection(host: Settings.shared.host, port: Settings.shared.port + 1)
                _unreliableConnection = connection
                connection.send(HelloMessage(message: "Hello from iOS over UDP!"))
                for try await receivedMessage in connection {
                    await handleMessage(receivedMessage, connection: connection)
                }
            } catch {
                log("Error: \(error.localizedDescription)")
            }
            _unreliableConnection = nil
            try? await Task.sleep(for: .seconds(5))
        }
    }

    private func handleMessage(_ receivedMessage: ReceivedJSONMessage, connection: any AsyncConnection) async {
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
