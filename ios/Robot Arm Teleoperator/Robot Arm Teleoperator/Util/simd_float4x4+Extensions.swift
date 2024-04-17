//
//  simd_float4x4+Extensions.swift
//  Robophone
//
//  Created by Bart Trzynadlowski on 4/7/24.
//

import simd

typealias Matrix4x4 = simd_float4x4

extension simd_float4x4 {
    static var identity: simd_float4x4 {
        return .init(diagonal: .one)
    }

    var position: simd_float3 {
        return simd_float3(x: self.columns.3.x, y: self.columns.3.y, z: self.columns.3.z)
    }

    var forward: simd_float3 {
        return simd_float3(x: self.columns.2.x, y: self.columns.2.y, z: self.columns.2.z)
    }

    var up: simd_float3 {
        return simd_float3(x: self.columns.1.x, y: self.columns.1.y, z: self.columns.1.z)
    }

    var right: simd_float3 {
        return simd_float3(x: self.columns.0.x, y: self.columns.0.y, z: self.columns.0.z)
    }

    init(translation: simd_float3, rotation: simd_quatf, scale: simd_float3) {
        let rotationMatrix = simd_matrix4x4(rotation)
        let scaleMatrix = simd_float4x4(diagonal: simd_float4(scale, 1.0))
        let translationMatrix = simd_float4x4(
        [
            simd_float4(x: 1, y: 0, z: 0, w: 0),
            simd_float4(x: 0, y: 1, z: 0, w: 0),
            simd_float4(x: 0, y: 0, z: 1, w: 0),
            simd_float4(translation, 1)
        ])
        let trs = translationMatrix * rotationMatrix * scaleMatrix
        self.init(columns: trs.columns)
    }
}

extension simd_float4x4: Codable {
    public init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        let values = try container.decode([Float].self)

        guard values.count == 16 else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "simd_float4x4 must have 16 values")
        }
        self.init(columns: (
                simd_float4(values[0], values[1], values[2], values[3]),
                simd_float4(values[4], values[5], values[6], values[7]),
                simd_float4(values[8], values[9], values[10], values[11]),
                simd_float4(values[12], values[13], values[14], values[15])
            )
        )
    }
    
    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode([
                columns.0.x, columns.0.y, columns.0.z, columns.0.w,
                columns.1.x, columns.1.y, columns.1.z, columns.1.w,
                columns.2.x, columns.2.y, columns.2.z, columns.2.w,
                columns.3.x, columns.3.y, columns.3.z, columns.3.w
            ]
        )
    }
}
