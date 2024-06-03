//
//  ServerConfigurationSheetView.swift
//  Robot Arm Teleoperator
//
//  Created by Bart Trzynadlowski on 6/2/24.
//

import SwiftUI

struct ServerConfigurationSheetView: View {
    @Binding var isShowing: Bool

    @State private var _host: String = Settings.shared.host
    @State private var _port: String = "\(Settings.shared.port)"

    var body: some View {
        VStack {
            Text("Server Endpoint")
                .font(.headline)
                .padding()

            HStack {
                Text("Host:")
                TextField("Host address", text: $_host)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            .padding()

            HStack {
                Text("Port:")
                TextField("Port", text: $_port)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .keyboardType(.numberPad)
            }
            .padding()

            Button(action: {
                isShowing = false
                Settings.shared.setPort(_port)
                Settings.shared.setHost(_host)
            }) {
                Text("OK")
                    .padding()
                    .background(Color.accentColor)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .padding()

            Spacer()
        }
        .padding()
    }
}

#Preview {
    ServerConfigurationSheetView(
        isShowing: .constant(true)
    )
}
