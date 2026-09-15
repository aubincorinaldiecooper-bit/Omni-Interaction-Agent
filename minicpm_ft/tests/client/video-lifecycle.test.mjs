// What the page does to the user's screen capture when things go wrong.
//
// The question behind every case here is the one from the audit: the capture
// was working, so did a transport problem stop it anyway?
import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  createPage,
  firstSentFrameId,
  makeStream,
  makeTrack,
  settle,
  shareScreen,
  SCREEN_CONFIG,
} from './harness.mjs';

test('sharing a screen reaches a live channel', async () => {
  const page = createPage();
  try {
    const { track, socket } = await shareScreen(page);
    assert.equal(track.stopCalls, 0);
    assert.equal(page.mediaDevices.displayMediaCalls, 1);
    assert.equal(socket.readyState, 1);
    assert.match(page.state(), /screen/);
  } finally {
    page.cleanup();
  }
});

test('a screen socket closed by the server does not stop the capture', async () => {
  const page = createPage();
  try {
    const { track, socket } = await shareScreen(page);

    socket.dropFromServer();
    await settle();

    // The one assertion this whole change exists for.
    assert.equal(track.stopCalls, 0, 'transport loss must not stop capture');
    assert.equal(track.readyState, 'live');
    assert.equal(
      page.mediaDevices.displayMediaCalls,
      1,
      'the stream is reused, so no second permission prompt'
    );
    assert.equal(page.select.value, 'screen', 'the choice is kept');
    assert.match(page.state(), /reconnecting/i);
  } finally {
    page.cleanup();
  }
});

test('the channel comes back on the same stream', async () => {
  const page = createPage();
  try {
    const { track, socket } = await shareScreen(page);
    const socketsBefore = page.sockets.length;

    socket.dropFromServer();
    await settle();
    // Wait out the first backoff step.
    await new Promise((resolve) => setTimeout(resolve, 700));

    // The retry renegotiates the mode on the existing stream.
    page.video.handleServerEvent({
      type: 'media.mode.done',
      video: true,
      session_id: 's1',
      screen: SCREEN_CONFIG,
      warnings: [],
      estimated_tokens_per_unit: 77,
    });
    await settle();

    assert.ok(page.sockets.length > socketsBefore, 'a new channel was opened');
    const next = page.sockets[page.sockets.length - 1];
    next.deliver({ type: 'screen.ready' });
    await settle();
    next.deliver({
      type: 'screen.frame.accepted',
      frame_id: firstSentFrameId(next),
    });
    await settle();

    assert.equal(track.stopCalls, 0);
    assert.equal(page.mediaDevices.displayMediaCalls, 1);
  } finally {
    page.cleanup();
  }
});

test('the browser stop-sharing control does end the capture', async () => {
  const page = createPage();
  try {
    const { track } = await shareScreen(page);

    track.fireEnded();
    await settle();

    assert.equal(track.stopCalls, 1, 'this is the one path that stops capture');
    assert.equal(page.select.value, '', 'and the only one that clears the choice');
    assert.equal(page.nodes.videoState.textContent, 'Off');
  } finally {
    page.cleanup();
  }
});

test('an explicit stop releases the track exactly once', async () => {
  const page = createPage();
  try {
    const { track } = await shareScreen(page);

    await page.video.stop();
    await settle();

    assert.equal(track.stopCalls, 1);
    assert.equal(page.nodes.videoState.textContent, 'Off');
  } finally {
    page.cleanup();
  }
});

test('a session drop keeps the share when the caller asks to keep it', async () => {
  const page = createPage();
  try {
    const { track, socket } = await shareScreen(page);

    // What cleanupAfterClose does for a close the user did not ask for.
    await page.video.stop({ keepCapture: true });
    await settle();

    assert.equal(track.stopCalls, 0, 'the share survives a duplex drop');
    assert.equal(track.readyState, 'live');
    assert.equal(socket.readyState, 3, 'but its channel is closed');
    assert.ok(page.video.isCapturing());
  } finally {
    page.cleanup();
  }
});

test('going idle detaches the channel without ending the share', async () => {
  const page = createPage();
  try {
    const { track } = await shareScreen(page);

    page.video.setSessionState('idle');
    await settle();

    assert.equal(track.stopCalls, 0);
    assert.ok(page.video.isCapturing());
    assert.match(page.state(), /sharing/i);
  } finally {
    page.cleanup();
  }
});

test('a withheld first-frame ack retries instead of stopping the share', async () => {
  const page = createPage();
  try {
    const track = makeTrack();
    page.mediaDevices.nextStream = makeStream(track);
    page.select.value = 'screen';
    page.video.setSessionState('active');

    const attached = page.video.attach(page.host);
    await settle();
    page.video.handleServerEvent({
      type: 'media.mode.done',
      video: true,
      session_id: 's1',
      screen: SCREEN_CONFIG,
      warnings: [],
      estimated_tokens_per_unit: 77,
    });
    await settle();
    page.sockets[page.sockets.length - 1].deliver({ type: 'screen.ready' });
    await settle();

    // Never acknowledge the frame. The 5s timeout should retry, not release.
    await new Promise((resolve) => setTimeout(resolve, 5400));
    await attached;

    assert.equal(track.stopCalls, 0, 'a slow ack is not a capture failure');
    assert.equal(page.mediaDevices.displayMediaCalls, 1);
  } finally {
    page.cleanup();
  }
});
