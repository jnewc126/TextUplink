import Foundation

/// WebSocket link to the Mac emotion server (server/cue_server.py).
final class CueSocket: NSObject, URLSessionWebSocketDelegate {
    var onMessage: (([String: Any]) -> Void)?
    var onState: ((Bool, String?) -> Void)?

    private var task: URLSessionWebSocketTask?
    private lazy var session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)

    func connect(to address: String) {
        disconnect()
        let trimmed = address.trimmingCharacters(in: .whitespaces)
        let withPort = trimmed.contains(":") ? trimmed : "\(trimmed):8765"
        guard let url = URL(string: "ws://\(withPort)") else { onState?(false, "Bad address"); return }
        let task = session.webSocketTask(with: url)
        self.task = task
        task.resume()
        receive(task)
    }

    func disconnect() {
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
    }

    func send(audio: Data) {
        task?.send(.data(audio)) { _ in }
    }

    func send(json: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: json), let text = String(data: data, encoding: .utf8) else { return }
        task?.send(.string(text)) { _ in }
    }

    private func receive(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self] result in
            guard let self, task === self.task else { return }
            switch result {
            case .success(.string(let text)):
                if let data = text.data(using: .utf8),
                   let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    self.onMessage?(object)
                }
                self.receive(task)
            case .success:
                self.receive(task)
            case .failure(let error):
                self.onState?(false, error.localizedDescription)
            }
        }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        if webSocketTask === task { onState?(true, nil) }
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask,
                    didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        if webSocketTask === task { onState?(false, "Server closed the connection") }
    }
}
