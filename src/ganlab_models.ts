import * as tf from '@tensorflow/tfjs-core';

// Hack to prevent error when using grads (doesn't allow this in model).
let dVariables: tf.Variable[];
let numDiscriminatorLayers: number;
let batchSize: number;

export class GANLabModel {
  dVariables: tf.Variable[];
  gVariables: tf.Variable[];

  dOptimizer: tf.Optimizer;
  gOptimizer: tf.Optimizer;
  lossType: string;

  constructor(
    private noiseSize: number,
    private numGeneratorLayers: number,
    private numDiscriminatorLayers: number,
    private numGeneratorNeurons: number,
    private numDiscriminatorNeurons: number,
    private batchSize: number,
    lossType: string
  ) { }

  initializeModelVariables() {
    if (this.dVariables) {
      this.dVariables.forEach((v: tf.Tensor) => v.dispose());
    }
    if (this.gVariables) {
      this.gVariables.forEach((v: tf.Tensor) => v.dispose());
    }
    // Filter variable nodes for optimizers.
    this.dVariables = [];
    this.gVariables = [];

    // Generator.
    const gfc0W = tf.variable(
      tf.randomNormal(
        [this.noiseSize, this.numGeneratorNeurons], 0, 1.0 / Math.sqrt(2)));
    const gfc0B = tf.variable(
      tf.zeros([this.numGeneratorNeurons]));

    this.gVariables.push(gfc0W);
    this.gVariables.push(gfc0B);

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = tf.variable(
        tf.randomNormal(
          [this.numGeneratorNeurons, this.numGeneratorNeurons], 0,
          1.0 / Math.sqrt(this.numGeneratorNeurons)));
      const gfcB = tf.variable(tf.zeros([this.numGeneratorNeurons]));

      this.gVariables.push(gfcW);
      this.gVariables.push(gfcB);
    }

    const gfcLastW = tf.variable(
      tf.randomNormal(
        [this.numGeneratorNeurons, 2], 0,
        1.0 / Math.sqrt(this.numGeneratorNeurons)));
    const gfcLastB = tf.variable(tf.zeros([2]));

    this.gVariables.push(gfcLastW);
    this.gVariables.push(gfcLastB);

    // Discriminator.
    const dfc0W = tf.variable(
      tf.randomNormal(
        [2, this.numDiscriminatorNeurons], 0, 1.0 / Math.sqrt(2)),
      true);
    const dfc0B = tf.variable(tf.zeros([this.numDiscriminatorNeurons]));

    this.dVariables.push(dfc0W);
    this.dVariables.push(dfc0B);

    for (let i = 0; i < this.numDiscriminatorLayers; ++i) {
      const dfcW = tf.variable(
        tf.randomNormal(
          [this.numDiscriminatorNeurons, this.numDiscriminatorNeurons], 0,
          1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
      const dfcB = tf.variable(tf.zeros([this.numDiscriminatorNeurons]));

      this.dVariables.push(dfcW);
      this.dVariables.push(dfcB);
    }

    const dfcLastW = tf.variable(
      tf.randomNormal(
        [this.numDiscriminatorNeurons, 1], 0,
        1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
    const dfcLastB = tf.variable(tf.zeros([1]));

    this.dVariables.push(dfcLastW);
    this.dVariables.push(dfcLastB);

    // Hack to prevent error when using grads (doesn't allow this in model).
    dVariables = this.dVariables;
    numDiscriminatorLayers = this.numDiscriminatorLayers;
    batchSize = this.batchSize;
  }

  async loadPretrainedWeights(loadedModel: tf.io.ModelArtifacts) {
    const decoded = tf.io.decodeWeights(
      loadedModel.weightData, loadedModel.weightSpecs);

    this.dVariables.forEach((v: tf.Variable, i) => {
      v.assign(decoded[`d-${i}`]);
    });
    this.gVariables.forEach((v: tf.Variable, i) => {
      v.assign(decoded[`g-${i}`]);
    });

    dVariables = this.dVariables;
  }

  generator(noiseTensor: tf.Tensor2D): tf.Tensor2D {
    const gfc0W = this.gVariables[0] as tf.Tensor2D;
    const gfc0B = this.gVariables[1];

    let network = noiseTensor.matMul(gfc0W)
      .add(gfc0B)
      .relu();

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = this.gVariables[2 + i * 2] as tf.Tensor2D;
      const gfcB = this.gVariables[3 + i * 2];

      network = network.matMul(gfcW)
        .add(gfcB)
        .relu();
    }

    const gfcLastW =
      this.gVariables[2 + this.numGeneratorLayers * 2] as tf.Tensor2D;
    const gfcLastB =
      this.gVariables[3 + this.numGeneratorLayers * 2];

    const generatedTensor: tf.Tensor2D = network.matMul(gfcLastW)
      .add(gfcLastB)
      .tanh() as tf.Tensor2D;

    return generatedTensor;
  }

  discriminator(inputTensor: tf.Tensor2D): tf.Tensor1D {
    const dfc0W = /*this.*/dVariables[0] as tf.Tensor2D;
    const dfc0B = /*this.*/dVariables[1];

    let network = inputTensor.matMul(dfc0W)
      .add(dfc0B)
      .relu();

    for (let i = 0; i < /*this.*/numDiscriminatorLayers; ++i) {
      const dfcW = /*this.*/dVariables[2 + i * 2] as tf.Tensor2D;
      const dfcB = /*this.*/dVariables[3 + i * 2];

      network = network.matMul(dfcW)
        .add(dfcB)
        .relu();
    }
    const dfcLastW =
      /*this.*/dVariables[2 + /*this.*/numDiscriminatorLayers * 2] as
      tf.Tensor2D;
    const dfcLastB =
      /*this.*/dVariables[3 + /*this.*/numDiscriminatorLayers * 2];

    const predictionTensor: tf.Tensor1D =
      network.matMul(dfcLastW)
        .add(dfcLastB)
        .sigmoid()
        .reshape([/*this.*/batchSize]);

    return predictionTensor;
  }

  // Define losses.
  dLoss(truePred: tf.Tensor1D, generatedPred: tf.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return tf.add(
        truePred.sub(tf.scalar(1)).square().mean(),
        generatedPred.square().mean()
      ) as tf.Scalar;
    } else {
      return tf.add(
        truePred.log().mul(tf.scalar(0.95)).mean(),
        tf.sub(tf.scalar(1), generatedPred).log().mean()
      ).mul(tf.scalar(-1)) as tf.Scalar;
    }
  }

  gLoss(generatedPred: tf.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return generatedPred.sub(tf.scalar(1)).square().mean() as tf.Scalar;
    } else {
      return generatedPred.log().mean().mul(tf.scalar(-1)) as tf.Scalar;
    }
  }

  updateOptimizer(
    dOrG: string, optimizerType: string, learningRate: number) {
    if (optimizerType === 'Adam') {
      const beta1 = 0.9;
      const beta2 = 0.999;
      if (dOrG === 'D') {
        this.dOptimizer = tf.train.adam(learningRate, beta1, beta2);
      }
      if (dOrG === 'G') {
        this.gOptimizer = tf.train.adam(learningRate, beta1, beta2);
      }
    } else {
      if (dOrG === 'D') {
        this.dOptimizer = tf.train.sgd(learningRate);
      }
      if (dOrG === 'G') {
        this.gOptimizer = tf.train.sgd(learningRate);
      }
    }
  }
}
