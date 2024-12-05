/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import * as mpPose from '@mediapipe/pose';
import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

import npyjs from "npyjs";
tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`);

import * as posedetection from '@tensorflow-models/pose-detection';
import * as tf from '@tensorflow/tfjs-core';

import { setupStats } from './stats_panel';
import { Context } from './camera';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setBackendAndEnvFlags } from './util';
import { keypointsToNormalizedKeypoints } from '../../../../shared/calculators/keypoints_to_normalized_keypoints';
let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
const statusElement = document.getElementById('status');
let buffer_poses = [];
let classificationModel;
const window_size = 30; // 30 poses of 26 keypoints per second (30fps). model was trained with this value
let window_stride = 1;
let window_start = 0;
const total_keypoints = 26;
const pose_classes = ["backhand", "forehand", "neutral", "serve"];
let score_threshold = 90;
let sessionSimilarityBackhand;
let sessionSimilarityForehand;
let sessionSimilarityServe;
let training_embeddings_backhand;
let training_embeddings_forehand;
let training_embeddings_serve;
let training_embeddings;
let count_labels = 0;
const INTERVAL_BETWEEN_SHOTS = 1;
let last_shot_position = 0;
let total_shots = [0, 0, 0, 0];
let next_video = 0;
const currentUrl = window.location.href;
const detected_shot = document.getElementById('detected_shot');
const reference_similarity = document.getElementById('reference_similarity');
const total_backhands = document.getElementById('total_backhands');
const total_forehands = document.getElementById('total_forehands');
const total_serves = document.getElementById('total_serves');
const score_classifier = document.getElementById('score_classifier');
const total_neutral = document.getElementById('total_neutral');

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.PoseNet:
      return posedetection.createDetector(STATE.model, {
        quantBytes: 4,
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: { width: 500, height: 500 },
        multiplier: 0.75
      });
    case posedetection.SupportedModels.BlazePose:
      const runtime = STATE.backend.split('-')[0];
      if (runtime === 'mediapipe') {
        return posedetection.createDetector(STATE.model, {
          runtime,
          modelType: STATE.modelConfig.type,
          solutionPath: `https://cdn.jsdelivr.net/npm/@mediapipe/pose@${mpPose.VERSION}`
        });
      } else if (runtime === 'tfjs') {
        return posedetection.createDetector(
          STATE.model, { runtime, modelType: STATE.modelConfig.type });
      }
    case posedetection.SupportedModels.MoveNet:
      const modelType = STATE.modelConfig.type == 'lightning' ?
        posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING :
        posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      return posedetection.createDetector(STATE.model, { modelType });
  }
}

async function checkGuiUpdate() {
  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    detector.dispose();

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    detector = await createDetector(STATE.model);
    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  // FPS only counts the time it takes to finish estimatePoses.
  beginEstimatePosesStats();

  const poses = await detector.estimatePoses(
    camera.video,
    { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });

  endEstimatePosesStats();

  camera.drawCtx();

  if (poses && poses.length > 0 && "keypoints" in poses[0]) buffer_poses.push(poses[0]);

  // The null check makes sure the UI is not in the middle of changing to a
  // different model. If during model change, the result is from an old
  // model, which shouldn't be rendered.
  if (poses.length > 0 && !STATE.isModelChanged) {
    camera.drawResults(poses);

    if (document.querySelector('input[name="modeuse"]:checked').value === "validation") return;

    //    if (buffer_poses.length > window_start + window_size && buffer_poses.length >= last_shot_position + INTERVAL_BETWEEN_SHOTS) {
    if (buffer_poses.length > window_start + window_size && buffer_poses.length >= last_shot_position + INTERVAL_BETWEEN_SHOTS) {
      let window_buffer = [];
      window_buffer = buffer_poses.slice(window_start, window_start + window_size);
      window_start += window_stride;
      // normalize the keypoints before sending to the models
      window_buffer = window_buffer.map(pose => keypointsToNormalizedKeypoints(pose.keypoints, { height: camera.video.height, width: camera.video.width }));
      //console.log("Mandando a calcular. Total:", window_buffer.length, " - Values:", window_buffer, "Stride:", window_stride);
      compute_distance_similiarity(window_buffer);
      last_shot_position = buffer_poses.length;
    } else {
      console.log("Waiting for frame...");
    }
  }
}

async function saveJSON(data, shot) {
  const jsonStr = JSON.stringify(data, null, 2); // Convert data to JSON string
  count_labels++;
  try {
    const response = await fetch('http://2080.local:5000', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Player': document.getElementById('videofile').files[0].name,
        'Shot': shot + "_" + count_labels
      },
      body: jsonStr
    });

    if (response.ok) {
      const result = await response.text();
      console.log(result);
    } else {
      console.log('Failed to upload file.');
    }
  } catch (error) {
    console.error('Error:', error);
    console.log('An error occurred while uploading the file.');
  }

}

// Function to save the current pose
function savePose(poses, key) {
  // Implement your logic to save the pose
  console.log('Pose saved:', poses, key);
  if (key === 'ArrowUp') key = "serve";
  if (key === 'ArrowRight') key = "forehand";
  if (key === 'ArrowLeft') key = "backhand";
  if (key === 'ArrowDown') key = "neutral";
  saveJSON(poses, key);
}

// Function to handle keydown events
function handleKeyDown(event) {


  if (document.querySelector('input[name="modeuse"]:checked').value === "inference") return;

  if (event.key === 'ArrowUp' || event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowDown') {
    // Get the current pose (assuming buffer_poses contains the latest poses)
    let window_buffer = buffer_poses.slice(buffer_poses.length - window_size, buffer_poses.length);
    window_buffer = window_buffer.map(pose => keypointsToNormalizedKeypoints(pose.keypoints, { height: camera.video.height, width: camera.video.width }));

    if (window_buffer.length === window_size) {
      savePose(window_buffer, event.key);
    }
  }
  event.stopPropagation();
  event.preventDefault();
}

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  URL.revokeObjectURL(camera.video.currentSrc);

  if (event.target.files) {
    const file = event.target.files[0];
    camera.source.src = URL.createObjectURL(file);
  } else {
    next_video++;
    if (next_video == 6) next_video = 5;
    camera.source.src = new URL(`/videos/${next_video}.mp4`, currentUrl).href;
  }
  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;

  statusElement.innerHTML = 'Video is loaded.';
  run();
}
/*  */
async function runFrame() {
  await checkGuiUpdate();
  if (video.paused) {
    // video has finished.
    camera.mediaRecorder.stop();
    camera.clearCtx();
    camera.video.style.visibility = 'visible';
    return;
  }
  await renderResult();
  rafId = requestAnimationFrame(runFrame);
}

async function run() {
  statusElement.innerHTML = 'Warming up model.';
  total_shots = [0, 0, 0, 0];
  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
      tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');

    await detector.estimatePoses(
      warmUpTensor,
      { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
    warmUpTensor.dispose();
    statusElement.innerHTML = 'Model is warmed up.';
  }

  camera.video.style.visibility = 'hidden';
  video.pause();
  video.currentTime = 0;
  video.play();
  camera.mediaRecorder.start();

  await new Promise((resolve) => {
    camera.video.onseeked = () => {
      resolve(video);
    };
  });

  await runFrame();
}

async function loadNpy(url) {
  let n = new npyjs();
  let data = await n.load(url);
  return data;
}

// TODO: Already normalize the training embeddings on loading
function l2Normalization(matrix) {
  let norm = math.matrix();
  for (let i = 0; i < matrix.size()[0]; i++) {
    const row = math.subset(matrix, math.index(i, math.range(0, matrix.size()[1])));
    let squares = row.map(vector => vector * vector);
    let sum = math.sum(squares);
    norm = math.concat(norm, [sum]);
  }
  norm = math.map(norm, math.sqrt);
  norm = math.reshape(norm, [norm.size()[0], 1]);
  const normalized_matrix = math.dotDivide(matrix, norm);
  return normalized_matrix;
}

function cosineSimilarity(vector1, vector2) {
  let Y_normalized = math.matrix(Array.from(vector2));
  Y_normalized = math.reshape(Y_normalized, [(vector2.length / 26), 26]);
  Y_normalized = l2Normalization(Y_normalized);

  let X_normalized = math.matrix(Array.from(vector1));
  X_normalized = math.reshape(X_normalized, [1, 26]);
  X_normalized = l2Normalization(X_normalized);

  let Y_normalized_tranpose = math.transpose(Y_normalized);
  const dotproduct = math.multiply(X_normalized, Y_normalized_tranpose);
  return dotproduct;
}

function convert_data_to_tensor(poses) {
  // convert data to both tf and onnx tensors
  let data = [];
  let ignore = ["left_eye", "right_eye", "left_ear", "right_ear"];
  for (const pose of poses) {
    let data_temp = [];
    for (const keypoint of pose) {
      if (ignore.includes(keypoint.name))
        continue;
      data_temp.push(keypoint.y);
      data_temp.push(keypoint.x);
    }
    data.push(data_temp);
  }

  let tensor_tf = tf.tensor([data], [1, window_size, total_keypoints]);
  let tensor_onnx = new ort.Tensor('float32', new Float32Array(data.flat()), [1, window_size, total_keypoints]);
  return { tensor_tf, tensor_onnx };
}

async function fetchCSV(url) {
  const response = await fetch(url);
  const csvText = await response.text();
  return csvText;
}

function parseCSV(csvText) {
  const lines = csvText.trim().split('\n');
  const headers = lines[0].split(',');
  const data = lines.slice(1).map(line => line.split(',').map(parseFloat));
  return data;
}

async function loadCSVIntoTensor(url) {
  const csvText = await fetchCSV(url);
  const data = parseCSV(csvText);
  const tensor_tf = tf.tensor(data);
  const tensor_onnx = new ort.Tensor('float32', new Float32Array(data.flat()), [30, 26]);
  return { tensor_tf, tensor_onnx };
}

async function compute_distance_similiarity(poses) {
  // the model expect 30 poses of 26 keypoints
  // i.e., 1 sec of 30fps
  //console.log("Poses:", poses);
  //saveJSON(poses, "debug");
  let tensors = convert_data_to_tensor(poses);
  //let tensors = await loadCSVIntoTensor("http://2080.local:3000/app/test_shots/forehand_1.csv");
  //const tensor_tf = tensors.tensor_tf.reshape([1, 30, 26]);

  const prediction = classificationModel.predict(tensors.tensor_tf);
  //prediction.print();
  let predicted_class = tf.argMax(prediction, 1).dataSync()[0];
  let score = tf.max(prediction, 1).dataSync()[0] * 100
  let output_msg = "";
  total_shots[predicted_class] += 1;

  if (pose_classes[predicted_class] !== "neutral" && score > score_threshold) {
    //output_msg = "Classifier output: " + pose_classes[predicted_class] + " with a score of " + score + "%";
    //output_msg = "Backhand: " + total_shots[0] + " - Forehand: " + total_shots[1] + " - Neutral: " + total_shots[2] + " - Serve: " + total_shots[3];
    //output_msg += "<br>Classifier output: " + pose_classes[predicted_class] + " with a score of " + score + "%";
    //document.getElementById('shot').innerHTML = output_msg;

    //console.log(output_msg);
    const inputs = { input: tensors.tensor_onnx };

    let cosine_sim;

    if (pose_classes[predicted_class].includes('backhand')) {
      const input_embeddings = await sessionSimilarityBackhand.run(inputs);
      cosine_sim = cosineSimilarity(input_embeddings.output.data, training_embeddings_backhand.data);
    } else if (pose_classes[predicted_class].includes('forehand')) {
      const input_embeddings = await sessionSimilarityForehand.run(inputs);
      cosine_sim = cosineSimilarity(input_embeddings.output.data, training_embeddings_forehand.data);
    } else if (pose_classes[predicted_class].includes('serve')) {
      const input_embeddings = await sessionSimilarityServe.run(inputs);
      cosine_sim = cosineSimilarity(input_embeddings.output.data, training_embeddings_serve.data);
    } else if (pose_classes[predicted_class].includes('neutral')) {
      console.log("Neutral pose detected, skipping similarity calculation");
      return;
    }

    const average_cosine_sim = (math.mean(cosine_sim) * 100).toFixed(3);
    //const sim_msg = `Similarity with Alcaraz ${pose_classes[predicted_class]}: ${average_cosine_sim}`;
    //document.getElementById('shotsimilarity').innerHTML = sim_msg;
    reference_similarity.innerHTML = average_cosine_sim + " %";
    detected_shot.innerHTML = pose_classes[predicted_class].toUpperCase();
    score_classifier.innerHTML = score + " %";
    total_backhands.innerHTML = total_shots[0];
    total_forehands.innerHTML = total_shots[1];
    total_neutral.innerHTML = total_shots[2];
    total_serves.innerHTML = total_shots[3];
    //console.log("Average cosine similarity:", average_cosine_sim);
  }
}

async function app() {
  // Gui content will change depending on which model is in the query string.
  const urlParams = new URLSearchParams(window.location.search);
  if (!urlParams.has('model')) {
    alert('Cannot find model in the query string.');
    return;
  }

  await setupDatGui(urlParams);
  stats = setupStats();
  camera = new Context();

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  await tf.ready();
  detector = await createDetector();

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;
  await (async () => {
    const tfjsmodule = await import('@tensorflow/tfjs');
    classificationModel = await tfjsmodule.loadLayersModel('/models/tfjs/classification/357/model.json');
    // Use the module within this scope
    //console.log(tfjsmodule.version); // Example: Log TensorFlow.js version

  })();

  console.log("Model loaded");

  const strideInput = document.getElementById('stride');
  strideInput.onchange = () => {
    if (isFinite(strideInput.value) && parseInt(strideInput.value) > 0) {
      window_stride = parseInt(strideInput.value);
    } else {
      window_stride = 1;
      strideInput.value = "Wrong value. Autoset to 1";
    }
  }

  const scoreInput = document.getElementById('shot_score');
  scoreInput.onchange = () => {
    if (isFinite(scoreInput.value) && parseInt(scoreInput.value) > 0) {
      score_threshold = parseInt(scoreInput.value);
    } else {
      score_threshold = 90;
      scoreInput.value = "Wrong value. Autoset to 0.9";
    }
  }

  const randomVideo = document.getElementById('startButton');
  randomVideo.onclick = updateVideo;

  sessionSimilarityBackhand = await ort.InferenceSession.create('/models/encoders/backhandalcaraz003_encoder/model.onnx');
  training_embeddings_backhand = await loadNpy('/models/embeddings/backhandalcaraz003.encoder.embeddings.npy');

  sessionSimilarityForehand = await ort.InferenceSession.create('/models/encoders/forehandalcaraz003_encoder/model.onnx');
  training_embeddings_forehand = await loadNpy('/models/embeddings/forehandalcaraz003.encoder.embeddings.npy');

  sessionSimilarityServe = await ort.InferenceSession.create('/models/encoders/servealcaraz003_encoder/model.onnx');
  training_embeddings_serve = await loadNpy('/models/embeddings/servealcaraz003.encoder.embeddings.npy');

  // Add the keydown event listener
  document.addEventListener('keydown', handleKeyDown);

  // load random video
};

app();
