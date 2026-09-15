// A browser just real enough to run video.js.
//
// These tests are about one question: what does the page do to the user's
// screen capture when the connection misbehaves? So the DOM, the media
// devices and the WebSocket are all stubs that record what was asked of them.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
export const STATIC_DIR = join(here, '..', '..', 'mcpmft', 'infer', 'static');

function element(extra = {}) {
  return {
    hidden: false,
    textContent: '',
    disabled: false,
    addEventListener() {},
    removeEventListener() {},
    ...extra,
  };
}

export function makeTrack() {
  const listeners = new Map();
  return {
    kind: 'video',
    readyState: 'live',
    stopCalls: 0,
    stop() {
      this.stopCalls += 1;
      this.readyState = 'ended';
    },
    addEventListener(name, fn) {
      listeners.set(name, fn);
    },
    removeEventListener(name) {
      listeners.delete(name);
    },
    // Stand in for the browser's own "Stop sharing" control.
    fireEnded() {
      const fn = listeners.get('ended');
      if (fn) fn();
    },
  };
}

export function makeStream(track) {
  return {
    getTracks: () => [track],
    getVideoTracks: () => [track],
  };
}

export function createPage() {
  const sockets = [];
  const controls = [];
  const events = [];
  const timers = new Set();

  class FakeWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url) {
      this.url = String(url);
      this.readyState = FakeWebSocket.OPEN;
      this.bufferedAmount = 0;
      this.sent = [];
      this.onmessage = null;
      this.onclose = null;
      this.onerror = null;
      sockets.push(this);
    }

    send(payload) {
      this.sent.push(payload);
    }

    close() {
      if (this.readyState >= FakeWebSocket.CLOSING) return;
      this.readyState = FakeWebSocket.CLOSED;
      if (this.onclose) this.onclose({});
    }

    // Drive the socket from the test's side.
    deliver(message) {
      if (this.onmessage) this.onmessage({ data: JSON.stringify(message) });
    }

    dropFromServer() {
      this.readyState = FakeWebSocket.CLOSED;
      if (this.onclose) this.onclose({});
    }
  }

  const select = element({
    value: '',
    options: [],
    selectedOptions: [],
  });

  const canvas = element({
    width: 0,
    height: 0,
    getContext: () => ({ drawImage() {} }),
    toBlob(cb) {
      cb({
        size: 128,
        arrayBuffer: async () => new ArrayBuffer(128),
      });
    },
  });

  const source = element({
    srcObject: null,
    videoWidth: 640,
    videoHeight: 480,
    play: async () => {},
  });

  const nodes = {
    videoModeControl: element(),
    videoModeSelect: select,
    videoPreviewWrap: element(),
    videoPreview: canvas,
    videoNote: element(),
    videoSource: source,
    videoState: element(),
  };

  const store = new Map();
  const mediaDevices = {
    displayMediaCalls: 0,
    userMediaCalls: 0,
    nextStream: null,
    async getDisplayMedia() {
      this.displayMediaCalls += 1;
      return this.nextStream;
    },
    async getUserMedia() {
      this.userMediaCalls += 1;
      return this.nextStream;
    },
  };

  const sandbox = {
    console,
    Promise,
    Error,
    Map,
    Set,
    Array,
    Object,
    JSON,
    Number,
    Boolean,
    String,
    Math,
    Date,
    ArrayBuffer,
    URL,
    WebSocket: FakeWebSocket,
    navigator: { mediaDevices },
    document: { getElementById: (id) => nodes[id] || null },
  };

  sandbox.window = {
    location: { protocol: 'http:', href: 'http://localhost/' },
    localStorage: {
      getItem: (k) => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, String(v)),
    },
    setTimeout: (fn, ms) => {
      const handle = setTimeout(fn, ms);
      timers.add(handle);
      return handle;
    },
    clearTimeout: (handle) => {
      clearTimeout(handle);
      timers.delete(handle);
    },
    addEventListener() {},
    dispatchEvent() {},
  };
  sandbox.setTimeout = sandbox.window.setTimeout;
  sandbox.clearTimeout = sandbox.window.clearTimeout;
  sandbox.globalThis = sandbox;

  vm.createContext(sandbox);
  vm.runInContext(readFileSync(join(STATIC_DIR, 'video.js'), 'utf8'), sandbox, {
    filename: 'video.js',
  });

  const host = {
    sendControl(payload) {
      controls.push(payload);
    },
    addEvent(message) {
      events.push(message);
    },
  };

  return {
    video: sandbox.window.GanderVideo,
    nodes,
    select,
    source,
    mediaDevices,
    sockets,
    controls,
    events,
    host,
    state: () => nodes.videoState.textContent,
    cleanup() {
      for (const handle of timers) clearTimeout(handle);
      timers.clear();
    },
  };
}

export const SCREEN_CONFIG = {
  path: '/ws/screen',
  token: 'screen-token',
  client_video: { enabled: true, sources: ['screen', 'camera'] },
  recommended_frame_rate: 1,
};

/** Drive a page from idle to "sharing, connected". */
export async function shareScreen(page) {
  const track = makeTrack();
  page.mediaDevices.nextStream = makeStream(track);
  page.select.value = 'screen';
  page.video.setSessionState('active');

  const attached = page.video.attach(page.host);
  await settle();

  // Answer the media.mode request the page just sent.
  page.video.handleServerEvent({
    type: 'media.mode.done',
    video: true,
    session_id: 's1',
    screen: SCREEN_CONFIG,
    warnings: [],
    estimated_tokens_per_unit: 77,
  });
  await settle();

  const socket = page.sockets[page.sockets.length - 1];
  socket.deliver({ type: 'screen.ready' });
  await settle();

  // Acknowledge the first frame so setup completes.
  const frameId = firstSentFrameId(socket);
  socket.deliver({ type: 'screen.frame.accepted', frame_id: frameId });
  await attached;
  await settle();
  return { track, socket };
}

export function firstSentFrameId(socket) {
  for (const payload of socket.sent) {
    if (typeof payload !== 'string') continue;
    const parsed = JSON.parse(payload);
    if (parsed.frame_id) return parsed.frame_id;
  }
  return null;
}

/** Let queued microtasks and zero-delay timers run. */
export async function settle(rounds = 6) {
  for (let i = 0; i < rounds; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 1));
  }
}
