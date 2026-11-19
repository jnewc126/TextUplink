# Glasses audio cue — first live test

The glasses microphone records the person in front of the wearer. The iPhone app
(`ios/GlassesCue`) streams that audio over Wi-Fi to a server on the Mac
(`server/cue_server.py`). The server runs emotion2vec+ with the Stage 1 gates and tells the
phone which word to play. The phone plays it through the glasses speaker.

```
glasses mic ──Bluetooth──▶ iPhone ──Wi-Fi WebSocket──▶ Mac: speech detection → emotion2vec+ → gate
glasses speaker ◀──Bluetooth── iPhone plays "happy" ◀──────── cue ◀────────────────┘
```

## Run it

1. **Mac:** one-time setup, then start the server.
   ```sh
   python3 -m venv .venv
   .venv/bin/pip install "numpy<2.1" torch torchaudio librosa soundfile "transformers>=4.57.6,<5" \
       "huggingface-hub>=0.34,<1" safetensors funasr==1.4.22 websockets
   .venv/bin/python scripts/provision_models.py emotion2vec
   .venv/bin/python scripts/provision_models.py emotion
   .venv/bin/python server/cue_server.py
   ```
   It prints the address to type into the phone, for example `192.168.1.173:8765`. If macOS
   asks whether Python may accept incoming connections, allow it.
2. **Glasses:** pair them with the iPhone in the Meta AI app. Check that iOS Settings ›
   Bluetooth shows them as connected.
3. **iPhone:** open `ios/GlassesCue.xcodeproj` in Xcode, choose the iPhone, and press Run.
   Allow microphone and local network access when asked.
4. In the app, enter the Mac address and tap **Start**. Under *Audio route*, the mic and
   speaker should both name the glasses. **Play test cue** checks the speaker on its own.
5. Have someone speak to the wearer. The cue is spoken after the speaker pauses.

The phone and the Mac must be on the same Wi-Fi network.

## How it decides

- **Speech detection:** a 20 ms frame counts as speech when it is 9 dB above the
  background level, which is the 10th percentile of the last 5 seconds.
- **Rolling estimate:** after 0.8 s of speech, the last 3 s are scored every 0.5 s.
- **Gates:** the team's Stage 1 gates: top score ≥ 0.70 and a lead of ≥ 0.15 over the
  next label. *Angry* must also be confirmed by the four-label comparison model.
  Neutral, Other and Unknown are never spoken.
- **Timing:** the latest passing estimate is spoken once the speaker has been quiet for
  0.4 s. The same word does not repeat within 4 s. While a cue plays, the phone stops
  sending microphone audio, so the cue is not analysed.

| Server option | Effect |
|---|---|
| `--cue-when now` | Speak as soon as an estimate passes, without waiting for a pause. This is faster but can be wrong: in testing, an angry clip read as *Happy 0.79* in its first second. |
| `--pause 0.6` | Wait longer before speaking, so the cue is less likely to talk over the speaker |
| `--reference all` | The team's full Stage 1 rule: also withhold when the comparison model disagrees on happy, sad or neutral. In testing, this blocked a clear happy clip. |
| `--min-score`, `--min-margin`, `--cooldown`, `--speech-db` | Tune the gates and the speech detection |
| `--wav clip.wav` | Replay a recording through the same live logic, without the phone |

## Measured so far (19 Nov, M5 Mac, recorded CREMA-D clips, no glasses)

| Clip | Estimates | Cue | Cue sent after speech ended |
|---|---|---|---|
| Actor 1001, happy | Happy 1.00, Happy 1.00 | happy ✓ | 400 ms |
| Actor 1001, angry | Happy 0.79, then Angry 0.98 | angry ✓ | 400 ms |

Model time was 70–150 ms per estimate on CPU. The 400 ms is the pause wait. Bluetooth
and Wi-Fi time are not included yet. The app shows the network round trip.

### First live test through the glasses (19 Nov)

The glasses were paired as a plain Bluetooth headset in iOS Settings, without the Meta AI
app. One person wore them and spoke, so the model scored the **wearer's own voice**.
The session lasted 56 seconds.

| Intended delivery | Estimates while speaking | Result |
|---|---|---|
| Excited / happy | Surprised 0.99 → Surprised 0.91 → Happy 0.97 → Happy 1.00 | "happy" ✓ |
| Flat | Neutral 0.95 → 0.97 → 0.88 | silent ✓ |
| Happy | Happy 1.00 → 0.76 | "happy" ✓ |
| Happy | Happy 0.98 | "happy" ✓ |
| Sad | Sad 0.51 → Neutral 0.57 → Neutral 0.66 | silent (withheld) |

Model time was 60–110 ms per estimate. Happy was easy to trigger. Sad sounded neutral
to the model. **Not tested yet:** a second person speaking to the wearer, which is the
real use case.

## Known limits

- **It cannot tell whose voice it hears.** The glasses microphones favour the wearer's
  mouth, so the wearer's own speech gets scored too. The gate's single-speaker check is
  assumed to pass.
- **Bluetooth headset audio is narrowband.** Expect lower accuracy than on clean clips.
- **The app must stay open** on the phone. It keeps the screen awake while it runs.
- Two recorded clips are a smoke test, not an accuracy measurement.
