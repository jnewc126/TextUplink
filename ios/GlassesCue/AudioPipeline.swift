import AVFoundation

/// Records the glasses microphone over Bluetooth, hands out 16 kHz mono Int16 chunks,
/// and plays short pre-rendered cue words back through the glasses speaker.
final class AudioPipeline {
    static let sampleRate = 16000.0

    var onChunk: ((Data) -> Void)?
    var onLevel: ((Float) -> Void)?

    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private let synth = AVSpeechSynthesizer()
    private let fallbackSynth = AVSpeechSynthesizer()
    private let outFormat = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: sampleRate, channels: 1, interleaved: true)!
    private let playFormat = AVAudioFormat(standardFormatWithSampleRate: 22050, channels: 1)!
    private var converter: AVAudioConverter?
    private var cues: [String: AVAudioPCMBuffer] = [:]
    private var muteUntil = Date.distantPast
    private let lock = NSLock()

    static let words: [String: String] = [
        "Happy": "happy", "Sad": "sad", "Angry": "angry", "Fearful": "scared",
        "Surprised": "surprised", "Disgusted": "disgusted",
    ]

    // MARK: Session and routing

    func configureSession() throws {
        let session = AVAudioSession.sharedInstance()
        var options: AVAudioSession.CategoryOptions = []
        if #available(iOS 26.0, *) { options.insert(.allowBluetoothHFP) } else { options.insert(.allowBluetooth) }
        try session.setCategory(.playAndRecord, mode: .default, options: options)
        try session.setPreferredSampleRate(Self.sampleRate)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
        if let glasses = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) {
            try session.setPreferredInput(glasses)
        }
    }

    var routeDescription: String {
        let route = AVAudioSession.sharedInstance().currentRoute
        let input = route.inputs.map { "\($0.portName) (\(Self.kind($0.portType)))" }.joined(separator: ", ")
        let output = route.outputs.map { "\($0.portName) (\(Self.kind($0.portType)))" }.joined(separator: ", ")
        return "Mic: \(input.isEmpty ? "none" : input)\nSpeaker: \(output.isEmpty ? "none" : output)"
    }

    var usingBluetooth: Bool {
        AVAudioSession.sharedInstance().currentRoute.inputs.contains { $0.portType == .bluetoothHFP }
    }

    private static func kind(_ type: AVAudioSession.Port) -> String {
        switch type {
        case .bluetoothHFP: return "Bluetooth headset"
        case .bluetoothA2DP: return "Bluetooth music"
        case .builtInMic: return "iPhone mic"
        case .builtInSpeaker: return "iPhone speaker"
        case .builtInReceiver: return "iPhone earpiece"
        default: return type.rawValue
        }
    }

    // MARK: Start / stop

    private var configObserver: NSObjectProtocol?
    var onRestart: ((String?) -> Void)?

    func start() throws {
        try configureSession()
        try startEngine()
        // Switching mic (iPhone -> glasses) resets the engine and silently removes the tap. Rebuild it.
        configObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            guard let self else { return }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                do { try self.startEngine(); self.onRestart?(nil) }
                catch { self.onRestart?(error.localizedDescription) }
            }
        }
    }

    private func startEngine() throws {
        if engine.isRunning { engine.stop() }
        let input = engine.inputNode
        let inFormat = input.outputFormat(forBus: 0)
        guard inFormat.sampleRate > 0 else { throw NSError(domain: "GlassesCue", code: 1, userInfo: [NSLocalizedDescriptionKey: "No microphone input available"]) }
        converter = AVAudioConverter(from: inFormat, to: outFormat)
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: AVAudioFrameCount(inFormat.sampleRate * 0.04), format: inFormat) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        if player.engine == nil { engine.attach(player) }
        engine.connect(player, to: engine.mainMixerNode, format: playFormat)
        engine.prepare()
        try engine.start()
        player.play()
    }

    func stop() {
        if let configObserver { NotificationCenter.default.removeObserver(configObserver) }
        configObserver = nil
        engine.inputNode.removeTap(onBus: 0)
        player.stop()
        engine.stop()
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let converter else { return }
        let ratio = Self.sampleRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
        guard let out = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var fed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, out.frameLength > 0, let samples = out.int16ChannelData?[0] else { return }
        let count = Int(out.frameLength)
        var sum: Float = 0
        for i in 0..<count { let v = Float(samples[i]) / 32768; sum += v * v }
        onLevel?(10 * log10(max(sum / Float(count), 1e-12)))
        lock.lock(); let muted = Date() < muteUntil; lock.unlock()
        if muted { return }  // don't send the cue word back to the server
        onChunk?(Data(bytes: samples, count: count * 2))
    }

    // MARK: Cue words

    /// Synthesize every cue word once up front so playing one later has no speech-engine delay.
    func prerenderCues(done: @escaping () -> Void) {
        let group = DispatchGroup()
        for (label, word) in Self.words {
            group.enter()
            var pieces: [AVAudioPCMBuffer] = []
            let utterance = AVSpeechUtterance(string: word)
            utterance.rate = 0.55
            var finished = false
            synth.write(utterance) { [weak self] buffer in
                guard let self, !finished, let pcm = buffer as? AVAudioPCMBuffer else { return }
                if pcm.frameLength == 0 {
                    finished = true
                    if let joined = self.join(pieces) { self.lock.lock(); self.cues[label] = joined; self.lock.unlock() }
                    group.leave()
                } else {
                    pieces.append(pcm)
                }
            }
        }
        group.notify(queue: .main, execute: done)
    }

    private func join(_ pieces: [AVAudioPCMBuffer]) -> AVAudioPCMBuffer? {
        guard let first = pieces.first, let conv = AVAudioConverter(from: first.format, to: playFormat) else { return nil }
        let total = pieces.reduce(0) { $0 + Double($1.frameLength) } * playFormat.sampleRate / first.format.sampleRate
        guard let out = AVAudioPCMBuffer(pcmFormat: playFormat, frameCapacity: AVAudioFrameCount(total + 1024)) else { return nil }
        var index = 0
        var error: NSError?
        conv.convert(to: out, error: &error) { _, status in
            if index < pieces.count { status.pointee = .haveData; index += 1; return pieces[index - 1] }
            status.pointee = .endOfStream
            return nil
        }
        return error == nil && out.frameLength > 0 ? out : nil
    }

    /// Plays the cue word now. Returns the word, or nil when the label has no cue.
    @discardableResult
    func play(label: String) -> String? {
        guard let word = Self.words[label] else { return nil }
        lock.lock(); let buffer = cues[label]; lock.unlock()
        let seconds: Double
        if let buffer, engine.isRunning {
            seconds = Double(buffer.frameLength) / buffer.format.sampleRate
            player.scheduleBuffer(buffer, at: nil, options: .interrupts)
            if !player.isPlaying { player.play() }
        } else {
            seconds = 0.8
            fallbackSynth.speak(AVSpeechUtterance(string: word))
        }
        lock.lock(); muteUntil = Date().addingTimeInterval(seconds + 0.25); lock.unlock()
        return word
    }
}
