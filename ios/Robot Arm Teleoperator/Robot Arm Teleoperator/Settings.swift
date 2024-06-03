//
//  Settings.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 6/2/24.
//

import Foundation

class Settings {
    static let shared = Settings()

    @Published private(set) var host: String = "192.168.0.100"
    @Published private(set) var port: UInt16 = 8000

    fileprivate init() {
        Self.registerDefaults()
        NotificationCenter.default.addObserver(self, selector: #selector(Self.onSettingsChanged), name: UserDefaults.didChangeNotification, object: nil)
        onSettingsChanged()
    }

    /// Sets the value of the server host name.
    /// - Parameter value: The new value.
    func setHost(_ value: String) {
        if host != value {
            host = value
            UserDefaults.standard.set(value, forKey: "host")
            print("[Settings] Set: host = \(host)")
        }
    }

    /// Sets the value of the server port (if valid).
    /// - Parameter value: The new value.
    func setPort(_ value: String) {
        if let newPort = UInt16(value) {
            if port != newPort {
                port = newPort
                UserDefaults.standard.set(value, forKey: "port")
                print("[Settings] Set: port = \(port)")
            }
        } else {
            print("[Settings] Error: Did not update port value because new value is out of range: \(value)")
        }
    }

    private static func getRootPListURL() -> URL? {
        guard let settingsBundle = Bundle.main.url(forResource: "Settings", withExtension: "bundle") else {
            print("[Settings] Could not find Settings.bundle")
            return nil
        }
        return settingsBundle.appendingPathComponent("Root.plist")
    }

    /// Sets the default values, if values do not already exist, for all settings from our Root.plist
    private static func registerDefaults() {
        guard let url = getRootPListURL() else {
            return
        }

        guard let settings = NSDictionary(contentsOf: url) else {
            print("[Settings] Couldn't find Root.plist in settings bundle")
            return
        }

        guard let preferences = settings.object(forKey: "PreferenceSpecifiers") as? [[String: AnyObject]] else {
            print("[Settings] Root.plist has an invalid format")
            return
        }

        var defaultsToRegister = [String: AnyObject]()
        for preference in preferences {
            if let key = preference["Key"] as? String,
               let value = preference["DefaultValue"] {
                print("[Settings] Registering default: \(key) = \(value.debugDescription ?? "<none>")")
                defaultsToRegister[key] = value as AnyObject
            }
        }

        UserDefaults.standard.register(defaults: defaultsToRegister)
    }

    @objc private func onSettingsChanged() {
        // Publish changes when settings have been edited
        let host = UserDefaults.standard.string(forKey: "host") ?? "192.168.0.100"
        if host != self.host {
            self.host = host
        }

        let portString = UserDefaults.standard.string(forKey: "port") ?? "8000"
        if let port = UInt16(portString) {
            if port != self.port {
                self.port = port
            }
        }
    }
}
