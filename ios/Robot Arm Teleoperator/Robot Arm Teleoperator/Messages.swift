//
//  Messages.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

struct HelloMessage: JSONMessage {
    let message: String
}

struct PoseStateMessage: JSONMessage {
    let gripperDeltaPosition: Vector3
    let gripperOpenAmount: Float
    let gripperRotateDegrees: Float
}
