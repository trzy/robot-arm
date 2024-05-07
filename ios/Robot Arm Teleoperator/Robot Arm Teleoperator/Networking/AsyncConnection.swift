//
//  Connection.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 5/6/24.
//

protocol AsyncConnection: AsyncSequence, CustomStringConvertible
    where AsyncIterator == AsyncThrowingStream<ReceivedJSONMessage, Error>.Iterator,
          Element == ReceivedJSONMessage {
    var isReliable: Bool {
        get
    }
    
    func send(_ message: JSONMessage)
    func close()

    func makeAsyncIterator() -> AsyncThrowingStream<ReceivedJSONMessage, any Error>.Iterator
}

