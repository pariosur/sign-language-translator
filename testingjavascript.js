import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://storage.googleapis.com/sign_language_translator/model.json');
