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

    @warn_unqualified_access
    func onHorizontalDrag(completion: @escaping (Float, Float) -> Void) -> some View {
        modifier(OnHorizontalDragGestureModifier(completion: completion))
    }

    @warn_unqualified_access
    func onVerticalDrag(completion: @escaping (Float, Float) -> Void) -> some View {
        modifier(OnVerticalDragGestureModifier(completion: completion))
    }
}

fileprivate struct OnTouchDownGestureModifier: ViewModifier {
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

fileprivate struct OnTouchUpGestureModifier: ViewModifier {
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

fileprivate struct OnHorizontalDragGestureModifier: ViewModifier {
    @State private var touching = false
    private let _completion: (Float, Float) -> Void

    func body(content: Content) -> some View {
        GeometryReader { geometry in
            content
                .simultaneousGesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            if !self.touching {
                                self.touching = true
                            }
                            let width = geometry.size.width
                            let startPct = Float(value.startLocation.x / width)
                            let currentPct = Float(value.location.x / width)
                            self._completion(startPct, currentPct)
                        }
                        .onEnded { _ in
                            self.touching = false
                        }
                )
        }
    }

    init(completion: @escaping (Float, Float) -> Void) {
        _completion = completion
    }
}

fileprivate struct OnVerticalDragGestureModifier: ViewModifier {
    @State private var touching = false
    private let _completion: (Float, Float) -> Void

    func body(content: Content) -> some View {
        GeometryReader { geometry in
            content
                .simultaneousGesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            if !self.touching {
                                self.touching = true
                            }
                            let height = geometry.size.height
                            let startPct = Float(value.startLocation.y / height)
                            let currentPct = Float(value.location.y / height)
                            self._completion(startPct, currentPct)
                        }
                        .onEnded { _ in
                            self.touching = false
                        }
                )
        }
    }

    init(completion: @escaping (Float, Float) -> Void) {
        _completion = completion
    }
}
