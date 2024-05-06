//
//  ConnectionError.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 5/6/24.
//

import Foundation
import Network

enum ConnectionError: Error {
    case invalidPort                    // connection cannot be opened because the port was invalid
    case failedToConnect                // connection failed to form
    case canceled                       // connection was canceled for an unknown reason
    case disconnected(error: NWError)   // disconnected while receiving data
}

extension ConnectionError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidPort:
            return "Invalid port number"
        case .failedToConnect:
            return "Failed to establish connection"
        case .canceled:
            return "Connection canceled"
        case .disconnected(let error):
            return "Disconnected from remote endpoint: \(error.localizedDescription)"
        }
    }
}
