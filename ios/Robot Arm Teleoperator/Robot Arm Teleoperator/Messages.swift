//
//  Messages.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

struct HelloMessage: JSONMessage {
    let message: String
}

struct PoseUpdateMessage: JSONMessage {
    let initialPose: Matrix4x4
    let pose: Matrix4x4
    let deltaPosition: Vector3
}

struct GripperOpenMessage: JSONMessage {
    let openAmount: Float
}

struct GripperRotateMessage: JSONMessage {
    let degrees: Float
}
