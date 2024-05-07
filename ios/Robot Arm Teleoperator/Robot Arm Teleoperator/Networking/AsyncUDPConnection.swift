//
//  AsyncUDPConnection.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 5/6/24.
//

import Foundation
import Network

class AsyncUDPConnection: AsyncConnection {
    private let _queue = DispatchQueue(label: "com.cambrianmoment.robot-arm-teleoperator.udp", qos: .default)
    private let _connection: NWConnection
    private var _connectContinuation: AsyncStream<AsyncUDPConnection>.Continuation?
    private var _receivedData: AsyncThrowingStream<ReceivedJSONMessage, Error>!
    private var _receivedDataContinuation: AsyncThrowingStream<ReceivedJSONMessage, Error>.Continuation?

    // MARK: AsyncConnection API

    init(host: String, port: UInt16) async throws {
        let host = NWEndpoint.Host(host)
        guard let port = NWEndpoint.Port(rawValue: port) else {
            throw ConnectionError.invalidPort
        }

        _connection = NWConnection(host: host, port: port, using: .udp)

        _receivedData = AsyncThrowingStream<ReceivedJSONMessage, Error> { [weak self] continuation in
            self?._queue.async { [weak self] in
                self?._receivedDataContinuation = continuation
            }
        }

        let connectStream = AsyncStream<AsyncUDPConnection> { [weak self] continuation in
            guard let self = self else {
                continuation.finish()
                return
            }
            _queue.async { [weak self] in
                guard let self = self else {
                    continuation.finish()
                    return
                }
                _connectContinuation = continuation
            }
        }

        log("Connecting to \(self)...")
        _connection.stateUpdateHandler = { [weak self] in
            self?.onState($0)
        }
        _connection.start(queue: _queue)

        var it = connectStream.makeAsyncIterator()
        guard let _ = await it.next() else {
            throw ConnectionError.failedToConnect
        }
    }

    var isReliable: Bool {
        return false
    }

    func send(_ message: JSONMessage) {
        send(message.serialize())
    }

    func close() {
        _connection.forceCancel()
    }

    // MARK: AsyncStream Conformance

    func makeAsyncIterator() -> AsyncThrowingStream<ReceivedJSONMessage, any Error>.Iterator {
        return _receivedData.makeAsyncIterator()
    }

    // MARK: Internal implementation

    private func onState(_ newState: NWConnection.State) {
        switch (newState) {
        case .ready:
            log("Connection \(self) established")
            _connectContinuation?.yield(self)   // indicates success
            _connectContinuation?.finish()
            _connectContinuation = nil
            receiveMessageHeader()

        case .cancelled:
            // If we were waiting to form a connection, indicate failure
            _connectContinuation?.finish()
            _connectContinuation = nil

            // If we were connected, throw disconnect error
            _receivedDataContinuation?.finish(throwing: ConnectionError.canceled)
            _receivedDataContinuation = nil

        case .failed(let error):
            log("Error: Connection \(self) failed: \(error.localizedDescription)")
            _connection.cancel()

        case .waiting(let error):
            log("Error: Connection \(self) could not be established: \(error.localizedDescription)")
            _connection.cancel()

        default:
            // Don't care
            break
        }
    }

    private func send(_ data: Data) {
        _connection.send(content: data, completion: .idempotent)
        print("Sent \(data.count) bytes")
    }

    private func send(_ data: Data, completion: ((NWError?) -> Void)?) {
        _connection.send(content: data, completion: .contentProcessed(completion ?? { _ in }))
    }

    private func receiveMessageHeader() {
        let headerSize = 4  // header is simply 4 bytes of total payload length (little endian)
        _connection.receive(minimumIncompleteLength: headerSize, maximumLength: headerSize) { [weak self] (content: Data?, _: NWConnection.ContentContext?, _: Bool, error: NWError?) in
            guard let self = self else { return }

            if let error = error {
                _receivedDataContinuation?.finish(throwing: ConnectionError.disconnected(error: error))
                _receivedDataContinuation = nil
            }

            var bodySize: Int = 0
            if let content = content {
                // Extract total message length
                let totalSize = UInt32(littleEndian: content.withUnsafeBytes { $0.load(as: UInt32.self) })
                if totalSize < headerSize || totalSize > Int.max {
                    // Size must at least be equal to header
                    log("Error: Received message with invalid header")
                    return
                }

                bodySize = Int(totalSize) - headerSize
            }

            receiveMessageBody(bodySize: bodySize)
        }
    }

    private func receiveMessageBody(bodySize: Int) {
        // 0-length messages are a special case
        if bodySize == 0 {
            // Always succeeds. Because message ID is contained in body, we have nothing to notify
            // delegate of and may simply proceed to the next message
            receiveMessageHeader()
            return
        }

        // Message has body
        _connection.receive(minimumIncompleteLength: bodySize, maximumLength: bodySize) { [weak self] (content: Data?, _: NWConnection.ContentContext?, _: Bool, error: NWError?) in
            guard let self = self else { return }

            if let error = error {
                _receivedDataContinuation?.finish(throwing: ConnectionError.disconnected(error: error))
                _receivedDataContinuation = nil
                return
            }

            if let content = content, let receivedMessage = JSONMessageDeserializer.deserialize(content) {
                _receivedDataContinuation?.yield(receivedMessage)
            }

            // Next message
            receiveMessageHeader()
        }
    }
}

extension AsyncUDPConnection: CustomStringConvertible {
    var description: String {
        return "udp://\(_connection.endpoint)"
    }
}

fileprivate func log(_ message: String) {
    print("[AsyncUDPConnection] \(message)")
}
