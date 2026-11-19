import SwiftUI

struct ContentView: View {
    @StateObject private var model = CueModel()

    var body: some View {
        NavigationStack {
            Form {
                Section("Mac emotion server") {
                    TextField("e.g. 192.168.1.173", text: $model.serverAddress)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .disabled(model.running)
                    Label(model.status, systemImage: model.connected ? "checkmark.circle.fill" : "circle.dashed")
                        .foregroundStyle(model.connected ? .green : .secondary)
                    if let rtt = model.roundTripMs { Text("Network round trip: \(rtt) ms").font(.footnote).foregroundStyle(.secondary) }
                }

                Section("Audio route") {
                    Text(model.route.isEmpty ? "Tap Refresh after connecting the glasses" : model.route)
                        .font(.callout)
                    if !model.route.isEmpty && !model.bluetooth {
                        Text("The glasses are not the active mic. Check they're connected in Bluetooth settings, then Stop and Start again.")
                            .font(.footnote).foregroundStyle(.orange)
                    }
                    Button("Refresh", action: model.refreshRoute)
                    Button("Play test cue (\"happy\")", action: model.testCue).disabled(!model.cuesReady)
                }

                Section("Live") {
                    HStack {
                        Text(model.speech ? "Speech" : "Listening")
                            .foregroundStyle(model.speech ? .blue : .secondary)
                        Spacer()
                        ProgressView(value: Double(max(0, min(1, (model.levelDb + 70) / 60))))
                            .frame(width: 140)
                    }
                    Text("Last estimate: \(model.lastEstimate)")
                    if let ms = model.lastInferMs { Text("Model time: \(ms) ms").font(.footnote).foregroundStyle(.secondary) }
                }

                Section {
                    Button(model.running ? "Stop" : "Start") { model.running ? model.stop() : model.start() }
                        .font(.headline)
                        .frame(maxWidth: .infinity)
                        .disabled(!model.cuesReady)
                }

                Section("Cues spoken") {
                    if model.cues.isEmpty { Text("None yet").foregroundStyle(.secondary) }
                    ForEach(model.cues) { cue in
                        HStack {
                            Text(cue.word).bold()
                            Text(String(format: "%.2f", cue.score)).foregroundStyle(.secondary)
                            Spacer()
                            if let ms = cue.serverMs { Text("\(ms) ms after speech").font(.footnote) }
                            Text(cue.time, style: .time).font(.footnote).foregroundStyle(.secondary)
                        }
                    }
                }

                Section {
                    Text("Estimates are guesses about tone of voice, not facts about feelings. The app can't yet tell the wearer's voice from the other person's.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Glasses Cue")
            .onAppear(perform: model.refreshRoute)
        }
    }
}
