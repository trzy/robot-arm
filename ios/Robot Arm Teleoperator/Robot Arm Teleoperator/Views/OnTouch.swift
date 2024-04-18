//
//  OnTouch.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 4/17/24.
//

import SwiftUI

extension View {
    @warn_unqualified_access
    func onTouchDown(completion: @escaping () -> Void) -> some View {
        modifier(OnTouchDownGestureModifier(completion: completion))
    }

    @warn_unqualified_access
    func onTouchUp(completion: @escaping () -> Void) -> some View {
        modifier(OnTouchUpGestureModifier(completion: completion))
    }
}

struct OnTouchDownGestureModifier: ViewModifier {
    @State private var tapped = false
    private let _completion: () -> Void

    func body(content: Content) -> some View {
        content
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !self.tapped {
                            self.tapped = true
                            self._completion()
                        }
                    }
                    .onEnded { _ in
                        self.tapped = false
                    }
            )
    }

    init(completion: @escaping () -> Void) {
        _completion = completion
    }
}

struct OnTouchUpGestureModifier: ViewModifier {
    @State private var tapped = false
    private let _completion: () -> Void

    func body(content: Content) -> some View {
        content
            .simultaneousGesture(
                DragGesture(minimumDistance: 0)
                    .onChanged { _ in
                        if !self.tapped {
                            self.tapped = true
                        }
                    }
                    .onEnded { _ in
                        self.tapped = false
                        self._completion()
                    }
            )
    }

    init(completion: @escaping () -> Void) {
        _completion = completion
    }
}
