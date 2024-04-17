//
//  simd_float3+Extensions.swift
//  Robophone
//
//  Created by Bart Trzynadlowski on 4/7/24.
//

import simd

typealias Vector3 = simd_float3

extension simd_float3 {
    static var forward: simd_float3 {
        return simd_float3(x: 0, y: 0, z: 1)
    }

    static var up: simd_float3 {
        return simd_float3(x: 0, y: 1, z: 0)
    }

    static var right: simd_float3 {
        return simd_float3(x: 1, y: 0, z: 0)
    }

    static func dot(_ u: simd_float3, _ v: simd_float3) -> Float {
        return simd_dot(u, v)
    }

    static func angle(_ u: simd_float3, _ v: simd_float3) -> Float {
        return acos(Vector3.dot(u, v) / (u.magnitude * v.magnitude)) * .rad2Deg
    }

    static func signedAngle(from u: simd_float3, to v: simd_float3, axis: simd_float3) -> Float {
        let unsignedAngle = Vector3.angle(u, v)
        let crossX = u.y * v.z - u.z * v.y
        let crossY = -(u.x * v.z - u.z * v.x)
        let crossZ = u.x * v.y - u.y * v.x
        let dot = (axis.x * crossX + axis.y * crossY + axis.z * crossZ)
        let sign: Float = dot >= 0 ? 1.0 : -1.0
        return unsignedAngle * sign
    }

    var normalized: simd_float3 {
        return simd_normalize(self)
    }

    var magnitude: Float {
        return simd_length(self)
    }

    var sqrMagnitude: Float {
        return simd_length_squared(self)
    }

    var distance: Float {
        return simd_length(self)
    }

    var xzProjected: simd_float3 {
        return simd_float3(x: self.x, y: 0, z: self.z)
    }
}
