//
//  Messages.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

struct HelloMessage: JSONMessage {
    let message: String
}

struct TransformMessage: JSONMessage {
    let matrix: Matrix4x4
}
