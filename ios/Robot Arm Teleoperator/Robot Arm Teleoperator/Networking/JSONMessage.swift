//
//  JSONMessage.swift
//  Robophone
//
//  Created by Bart Trzynadlowski on 3/31/24.
//
//  JSON messages with an __id field added automatically during serialization and size prefix.
//

import Foundation

struct ReceivedJSONMessage {
    let id: String
    let jsonData: Data
}

fileprivate struct IDField: Decodable {
    let __id: String
}

protocol JSONMessage: Codable {
}

fileprivate let _decoder = JSONDecoder()

extension JSONMessage {
    static var id: String {
        return String(describing: Self.self)
    }

    func serialize() -> Data {
        do {
            // Encode as JSON and replace the final '}' with ',"__id":"ClassName"}'
            var jsonData = try JSONEncoder().encode(self)
            if jsonData.count > 0 && jsonData[jsonData.count - 1] == Character("}").asciiValue! {
                if let extraData = "\"__id\":\"\(Self.id)\"}".data(using: .utf8) {
                    jsonData[jsonData.count - 1] = Character(",").asciiValue!
                    jsonData.append(extraData)
                }
            }

            // Add 4 byte size header
            if var totalSize = UInt32(exactly: 4 + jsonData.count) {
                var data = Data(capacity: Int(totalSize))
                withUnsafePointer(to: &totalSize) {
                    data.append(UnsafeBufferPointer(start: $0, count: 1))
                }
                data.append(jsonData)
                return data
            }
        } catch {
            print("[JSONMessage]: Serialization failed")
        }

        return Data()
    }

    static func deserialize(_ data: Data) -> ReceivedJSONMessage? {
        let decoder = JSONDecoder()
        do {
            let idField = try decoder.decode(IDField.self, from: data)
            return ReceivedJSONMessage(id: idField.__id, jsonData: data)
        } catch {
            print("[JSONMessage]: Deserialization failed")
            return nil
        }
    }

    static func decode<T>(_ receivedMessage: ReceivedJSONMessage, as type: T.Type) -> T? where T: JSONMessage {
        return try? _decoder.decode(type.self, from: receivedMessage.jsonData)
    }
}

/// Allows JSONMessage's static deserialize() method to be called. Swift does not permit static
/// methods to be called on the protocol metatype directly, hence this dummy concrete type.
struct JSONMessageDeserializer: JSONMessage {
}
