//
//  NonReactiveState.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/24/24.
//
//  Originally written by Tomáš Kafka:
//  https://tomaskafka.medium.com/improving-swiftui-performance-managing-view-state-without-unnecessary-redraws-1ea1399967fb
//

/// Wrapper type that can store a value and be used as a SwiftUI state property without triggering
/// any rendering updates. Very useful when needing to store stateful information about e.g. the
/// start of a drag gesture, such as its start time, without triggering SwiftUI.
///
/// Example usage:
/// ```
/// @State private var _initialGripperOpenValue = NonReactiveState<Float>(wrappedValue: 0)
/// ```
class NonReactiveState<T> {
    private var _value: T

    init(wrappedValue: T) {
        self._value = wrappedValue
    }

    var wrappedValue: T {
        get { return _value }
        set { _value = newValue }
    }
}
