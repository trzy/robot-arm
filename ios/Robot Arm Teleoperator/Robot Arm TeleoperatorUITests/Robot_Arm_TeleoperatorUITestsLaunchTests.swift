//
//  Robot_Arm_TeleoperatorUITestsLaunchTests.swift
//  Robot Arm TeleoperatorUITests
//
//  Created by Bart Trzynadlowski on 4/16/24.
//

import XCTest

final class Robot_Arm_TeleoperatorUITestsLaunchTests: XCTestCase {

    override class var runsForEachTargetApplicationUIConfiguration: Bool {
        true
    }

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testLaunch() throws {
        let app = XCUIApplication()
        app.launch()

        // Insert steps here to perform after app launch but before taking a screenshot,
        // such as logging into a test account or navigating somewhere in the app

        let attachment = XCTAttachment(screenshot: app.screenshot())
        attachment.name = "Launch Screen"
        attachment.lifetime = .keepAlways
        add(attachment)
    }
}
