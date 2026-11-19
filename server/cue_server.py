"""Live vocal-emotion cue server for the glasses audio test.

The iPhone app streams the glasses microphone here as 16 kHz mono 16-bit PCM over a
WebSocket. This server finds speech, re-scores the most recent speech with emotion2vec+
every half second, applies the team's Stage 1 gates, and tells the phone which word to
play in the glasses.

    .venv/bin/python server/cue_server.py                 # cue at the next pause (default)
    .venv/bin/python server/cue_server.py --cue-when now  # cue as soon as an estimate passes
    .venv/bin/python server/cue_server.py --wav clip.wav  # replay a file instead of the phone

Estimates are model guesses about vocal delivery, not facts about feelings.
"""
import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from textuplink.emotion import Emotion2Vec, gate_prediction  # noqa: E402

RATE = 16000
FRAME = 320                     # 20 ms analysis frame
PRE_ROLL = int(.2 * RATE)       # keep 200 ms before speech onset
MIN_SPEECH = .8                 # seconds of voiced frames before the first estimate
WINDOW = 3.0                    # seconds of audio given to the model
STEP = .5                       # re-score this often while someone is speaking
MAX_UTTERANCE = 8.0             # keep at most this much of one utterance


class VoiceModel(Emotion2Vec):
    """Team's emotion2vec+ wrapper, loaded without the Python 3.11-only checkpoint hash."""
    def __init__(self, path, device):
        from funasr import AutoModel
        logging.disable(logging.WARNING)  # FunASR logs every checkpoint key at INFO
        self.path = Path(path)
        if not (self.path / 'model.pt').is_file():
            sys.exit('Missing emotion2vec+. Run: .venv/bin/python scripts/provision_models.py emotion2vec')
        self.model = AutoModel(model=str(self.path), device=device, hub='hf', disable_update=True,
                               disable_pbar=True, disable_log=True, trust_remote_code=False,
                               ncpu=4, log_level='ERROR')


class ReferenceModel:
    """The four-label SUPERB model the team's gate uses to corroborate anger."""
    LABELS = {'neu': 'Neutral', 'hap': 'Happy', 'ang': 'Angry', 'sad': 'Sad'}

    def __init__(self, path):
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        if not (Path(path) / 'config.json').is_file():
            sys.exit('Missing reference model. Run: .venv/bin/python scripts/provision_models.py emotion')
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(path), local_files_only=True)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(str(path), local_files_only=True).eval()

    def estimate(self, y):
        import torch
        inputs = self.extractor(y, sampling_rate=RATE, return_tensors='pt')
        with torch.inference_mode():
            scores = self.model(**inputs).logits.softmax(dim=-1)[0]
        idx = int(scores.argmax())
        return self.LABELS.get(self.model.config.id2label[idx]), float(scores[idx])


class Scorer:
    def __init__(self, args):
        started = time.perf_counter()
        self.voice = VoiceModel(ROOT / 'models/emotion2vec_plus_large', args.device)
        self.reference = None if args.reference == 'off' else ReferenceModel(ROOT / 'models/emotion')
        self.args = args
        self.lock = threading.Lock()
        print(f'Models loaded in {time.perf_counter() - started:.1f} s', flush=True)
        self.score(np.random.default_rng(0).normal(0, .05, RATE * 2).astype(np.float32))  # warm up

    def score(self, y):
        with self.lock:
            started = time.perf_counter()
            scores = self.voice.distribution(y, sample_rate=RATE)
            top = max(scores, key=scores.get)
            check = self.reference and (self.args.reference == 'all' or top == 'Angry')
            ref_label, ref_score = self.reference.estimate(y) if check else (None, None)
            infer_ms = 1000 * (time.perf_counter() - started)
        decision = gate_prediction(
            scores, duration_s=len(y) / RATE, rms=float(np.sqrt(np.mean(y.astype(np.float64) ** 2))),
            clipping_fraction=float(np.mean(np.abs(y) >= .99)),
            one_speaker=True,  # not checked yet: the wearer's own voice is also picked up
            minimum_score=self.args.min_score, minimum_margin=self.args.min_margin,
            reference_label=ref_label, reference_score=ref_score)
        decision['infer_ms'] = infer_ms
        decision['reference'] = ref_label
        return decision


class Session:
    """Speech detection and cue timing for one connected phone."""
    def __init__(self, scorer, send, args):
        self.scorer, self.send, self.args = scorer, send, args
        self.pending = np.zeros(0, np.float32)
        self.history = []                    # recent frame levels for the noise floor
        self.pre = np.zeros(0, np.float32)
        self.reset()
        self.last_cue = {}                   # label -> time last spoken
        self.busy = False
        self.received = 0

    def reset(self):
        self.utterance = []
        self.utt_samples = 0
        self.speech_samples = 0
        self.silence_samples = 0
        self.scored_at = 0
        self.latest = None
        self.cued_this_utt = set()
        self.last_speech_at = None
        self.utt_id = getattr(self, 'utt_id', 0) + 1

    def floor_db(self):
        # Background level: a low percentile of the last 5 s, so it follows room noise but not speech.
        return float(np.percentile(self.history[-250:], 10)) if len(self.history) >= 25 else -60.

    async def feed(self, pcm):
        self.received += len(pcm)
        self.pending = np.concatenate([self.pending, pcm])
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            await self.frame(frame)

    async def frame(self, frame):
        db = 20 * np.log10(max(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2))), 1e-8))
        floor = self.floor_db()
        speech = db > max(floor + self.args.speech_db, -55.)
        self.history.append(db)
        del self.history[:-600]
        in_utt = self.utt_samples > 0
        if speech:
            if not in_utt:
                self.utterance = [self.pre.copy()]
                self.utt_samples = len(self.pre)
            self.utterance.append(frame)
            self.utt_samples += FRAME
            self.speech_samples += FRAME
            self.silence_samples = 0
            self.last_speech_at = time.monotonic()
        elif in_utt:
            self.utterance.append(frame)
            self.utt_samples += FRAME
            self.silence_samples += FRAME
        self.pre = np.concatenate([self.pre, frame])[-PRE_ROLL:]
        if self.utt_samples > MAX_UTTERANCE * RATE:
            y = np.concatenate(self.utterance)[-int(WINDOW * RATE):]
            self.utterance, self.utt_samples = [y], len(y)
            self.scored_at = 0
        if self.utt_samples == 0:
            if len(self.history) % 25 == 0:
                await self.status(db, floor, False)
            return
        if len(self.history) % 10 == 0:
            await self.status(db, floor, speech)
        pause = self.silence_samples >= self.args.pause * RATE
        enough = self.speech_samples >= MIN_SPEECH * RATE
        if enough and not self.busy and not pause and self.utt_samples - self.scored_at >= STEP * RATE:
            self.scored_at = self.utt_samples
            self.busy = True
            asyncio.get_running_loop().create_task(self.rolling(self.clip(), self.utt_id))
        if pause:
            if enough and self.latest is None and not self.busy:
                self.busy = True
                await self.rolling(self.clip(), self.utt_id)
            await self.maybe_cue(at_pause=True)
            self.reset()

    def clip(self):
        y = np.concatenate(self.utterance)
        trailing = self.silence_samples
        if trailing:
            y = y[:max(len(y) - trailing + int(.1 * RATE), RATE)]
        return y[-int(WINDOW * RATE):].astype(np.float32)

    async def rolling(self, y, utt_id):
        try:
            decision = await asyncio.get_running_loop().run_in_executor(None, self.scorer.score, y)
        except Exception as error:
            print(f'  scoring failed: {error}', flush=True)
            decision = None
        finally:
            self.busy = False
        if decision is None or utt_id != self.utt_id:
            return
        self.latest = decision
        print(f"  {len(y) / RATE:4.1f}s  {decision['top_label']:<9} {decision['top_score']:.2f}  "
              f"-> {decision['estimate']:<9} ({decision['infer_ms']:.0f} ms)"
              + ('' if not decision['reasons'] else '  withheld: ' + '; '.join(decision['reasons'])), flush=True)
        await self.send({'type': 'estimate', 'top': decision['top_label'], 'score': decision['top_score'],
                         'estimate': decision['estimate'], 'reasons': decision['reasons'],
                         'infer_ms': round(decision['infer_ms'])})
        if self.args.cue_when == 'now':
            await self.maybe_cue(at_pause=False)

    async def maybe_cue(self, at_pause):
        d = self.latest
        if d is None or d['abstained'] or d['estimate'] == 'Neutral':
            return
        label = d['estimate']
        now = time.monotonic()
        if label in self.cued_this_utt or now - self.last_cue.get(label, -1e9) < self.args.cooldown:
            return
        self.cued_this_utt.add(label)
        self.last_cue[label] = now
        since_speech = None if self.last_speech_at is None else 1000 * (now - self.last_speech_at)
        print(f'>>> CUE {label.upper()}  ({d["top_score"]:.2f})', flush=True)
        await self.send({'type': 'cue', 'label': label, 'score': d['top_score'], 'at_pause': at_pause,
                         'infer_ms': round(d['infer_ms']),
                         'since_last_speech_ms': None if since_speech is None else round(since_speech)})
        if at_pause:
            return
        self.reset()

    async def status(self, db, floor, speech):
        await self.send({'type': 'status', 'level_db': round(db, 1), 'floor_db': round(floor, 1),
                         'speech': bool(speech), 'received_s': round(self.received / RATE, 1)})


async def serve(args, scorer):
    import websockets

    async def handler(ws):
        peer = ws.remote_address[0] if ws.remote_address else '?'
        print(f'Phone connected from {peer}', flush=True)

        async def send(message):
            try:
                await ws.send(json.dumps(message))
            except Exception:
                pass

        session = Session(scorer, send, args)
        await send({'type': 'hello', 'cue_when': args.cue_when, 'pause_s': args.pause})
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    await session.feed(np.frombuffer(message, '<i2').astype(np.float32) / 32768.)
                else:
                    data = json.loads(message)
                    if data.get('type') == 'ping':
                        await send({'type': 'pong', 't': data.get('t')})
                    elif data.get('type') == 'cue_played':
                        session.reset()  # ignore the cue word echoing back into the mic
        finally:
            print(f'Phone disconnected after {session.received / RATE:.0f} s of audio', flush=True)

    addresses = []
    try:  # the address this Mac uses on the local network; nothing is sent
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(('10.255.255.255', 1))
        addresses.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass
    async with websockets.serve(handler, '0.0.0.0', args.port, max_size=2 ** 20, ping_interval=10):
        print(f'Listening on port {args.port}. In the phone app enter one of: '
              + ', '.join(f'{a}:{args.port}' for a in addresses or ['<this Mac IP>']), flush=True)
        await asyncio.Future()


async def replay(args, scorer):
    """Feed a recording through the same live logic, at real-time speed, for testing without the phone."""
    import librosa
    y, _ = librosa.load(args.wav, sr=RATE, mono=True)
    started = time.monotonic()

    async def send(message):
        if message['type'] == 'cue':
            print(f"    cue sent at {time.monotonic() - started:5.2f}s of playback "
                  f"({message['since_last_speech_ms']} ms after last speech)", flush=True)

    session = Session(scorer, send, args)
    chunk = int(.04 * RATE)
    for i in range(0, len(y) + RATE, chunk):
        part = y[i:i + chunk] if i < len(y) else np.zeros(chunk, np.float32)
        await session.feed(part.astype(np.float32))
        await asyncio.sleep(max(0., started + (i + chunk) / RATE - time.monotonic()))
    while session.busy:
        await asyncio.sleep(.05)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--port', type=int, default=8765)
    p.add_argument('--cue-when', choices=['pause', 'now'], default='pause',
                   help='pause: wait for the speaker to pause (default); now: speak as soon as an estimate passes')
    p.add_argument('--pause', type=float, default=.4, help='seconds of quiet that count as a pause (default 0.4)')
    p.add_argument('--speech-db', type=float, default=9., help='dB above the background level that counts as speech')
    p.add_argument('--min-score', type=float, default=.7, help="team gate: minimum top score (default 0.7)")
    p.add_argument('--min-margin', type=float, default=.15, help='team gate: minimum lead over the runner-up')
    p.add_argument('--cooldown', type=float, default=4., help='seconds before the same word can repeat')
    p.add_argument('--device', default='cpu', help='cpu (default) or mps')
    p.add_argument('--reference', choices=['anger', 'all', 'off'], default='anger',
                   help="comparison model: 'anger' checks only anger estimates (default); 'all' is the team's "
                        "full Stage 1 rule, which also withholds any disagreement; 'off' means anger never passes")
    p.add_argument('--wav', help='replay this audio file instead of listening for the phone')
    args = p.parse_args()
    scorer = Scorer(args)
    try:
        asyncio.run(replay(args, scorer) if args.wav else serve(args, scorer))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
