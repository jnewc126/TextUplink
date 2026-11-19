import AVFoundation
import SwiftUI

struct CueEvent: Identifiable {
    let id = UUID()
    let time = Date()
    let word: String
    let score: Double
    let serverMs: Int?       // server: last speech -> cue sent
    let roundTripMs: Int?    // network ping
}

@MainActor
final class CueModel: ObservableObject {
    @AppStorage("serverAddress") var serverAddress = ""
    @Published var running = false
    @Published var connected = false
    @Published var status = "Not started"
    @Published var route = ""
    @Published var bluetooth = false
    @Published var levelDb: Float = -90
    @Published var speech = false
    @Published var lastEstimate = "—"
    @Published var lastInferMs: Int?
    @Published var roundTripMs: Int?
    @Published var cues: [CueEvent] = []
    @Published var cuesReady = false

    private let audio = AudioPipeline()
    private let socket = CueSocket()
    private var pingTimer: Timer?
    private var routeObserver: NSObjectProtocol?

    init() {
        audio.onChunk = { [weak self] data in self?.socket.send(audio: data) }
        audio.onLevel = { [weak self] db in Task { @MainActor in self?.levelDb = db } }
        socket.onState = { [weak self] ok, error in
            Task { @MainActor in
                guard let self else { return }
                self.connected = ok
                if ok { self.status = "Connected to Mac" }
                else if self.running { self.status = "Disconnected: \(error ?? "unknown")" }
            }
        }
        socket.onMessage = { [weak self] message in Task { @MainActor in self?.handle(message) } }
        audio.onRestart = { [weak self] error in
            Task { @MainActor in
                guard let self else { return }
                self.refreshRoute()
                if let error { self.status = "Audio restart failed: \(error)" }
            }
        }
        audio.prerenderCues { [weak self] in self?.cuesReady = true }
        routeObserver = NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.refreshRoute() }
        }
    }

    func refreshRoute() {
        route = audio.routeDescription
        bluetooth = audio.usingBluetooth
    }

    func start() {
        guard !serverAddress.isEmpty else { status = "Enter the Mac address first"; return }
        AVAudioApplication.requestRecordPermission { [weak self] granted in
            Task { @MainActor in
                guard let self else { return }
                guard granted else { self.status = "Microphone permission denied. Allow it in Settings › GlassesCue."; return }
                do {
                    try self.audio.start()
                    self.socket.connect(to: self.serverAddress)
                    self.running = true
                    self.status = "Connecting…"
                    UIApplication.shared.isIdleTimerDisabled = true
                    self.refreshRoute()
                    self.pingTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
                        Task { @MainActor in self?.socket.send(json: ["type": "ping", "t": Date().timeIntervalSince1970]) }
                    }
                } catch {
                    self.status = "Audio failed: \(error.localizedDescription)"
                }
            }
        }
    }

    func stop() {
        pingTimer?.invalidate()
        socket.disconnect()
        audio.stop()
        running = false
        connected = false
        speech = false
        status = "Stopped"
        UIApplication.shared.isIdleTimerDisabled = false
        refreshRoute()
    }

    func testCue() {
        if !running { try? audio.configureSession(); refreshRoute() }
        audio.play(label: "Happy")
    }

    private func handle(_ m: [String: Any]) {
        switch m["type"] as? String {
        case "status":
            speech = m["speech"] as? Bool ?? false
        case "estimate":
            let top = m["top"] as? String ?? "?"
            let estimate = m["estimate"] as? String ?? "?"
            let score = m["score"] as? Double ?? 0
            lastEstimate = estimate == top ? "\(top) \(String(format: "%.2f", score))"
                                           : "\(estimate) (top: \(top) \(String(format: "%.2f", score)))"
            lastInferMs = m["infer_ms"] as? Int
        case "cue":
            guard let label = m["label"] as? String, let word = audio.play(label: label) else { return }
            socket.send(json: ["type": "cue_played", "label": label])
            cues.insert(CueEvent(word: word, score: m["score"] as? Double ?? 0,
                                 serverMs: m["since_last_speech_ms"] as? Int, roundTripMs: roundTripMs), at: 0)
            if cues.count > 30 { cues.removeLast() }
        case "pong":
            if let t = m["t"] as? Double { roundTripMs = Int((Date().timeIntervalSince1970 - t) * 1000) }
        default:
            break
        }
    }
}
