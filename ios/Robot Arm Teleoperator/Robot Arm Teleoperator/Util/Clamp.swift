//
//  Clamp.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/24/24.
//

func clamp<T>(_ value: T, min minValue: T, max maxValue: T) -> T where T: Comparable {
    return max(minValue, min(maxValue, value))
}
