import * as dl from 'deeplearn';

// Hack to prevent error when using grads (doesn't allow this in model).
let dVariables: dl.Variable[];
let numDiscriminatorLayers: number;
let batchSize: number;

export class GANLabModel {
  dVariables: dl.Variable[];
  gVariables: dl.Variable[];

  dOptimizer: dl.Optimizer;
  gOptimizer: dl.Optimizer;
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
      this.dVariables.forEach((v: dl.Tensor) => v.dispose());
    }
    if (this.gVariables) {
      this.gVariables.forEach((v: dl.Tensor) => v.dispose());
    }
    // Filter variable nodes for optimizers.
    this.dVariables = [];
    this.gVariables = [];

    // Generator.
    const gfc0W = dl.variable(
      dl.randomNormal(
        [this.noiseSize, this.numGeneratorNeurons], 0, 1.0 / Math.sqrt(2)));
    const gfc0B = dl.variable(
      dl.zeros([this.numGeneratorNeurons]));

    this.gVariables.push(gfc0W);
    this.gVariables.push(gfc0B);

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = dl.variable(
        dl.randomNormal(
          [this.numGeneratorNeurons, this.numGeneratorNeurons], 0,
          1.0 / Math.sqrt(this.numGeneratorNeurons)));
      const gfcB = dl.variable(dl.zeros([this.numGeneratorNeurons]));

      this.gVariables.push(gfcW);
      this.gVariables.push(gfcB);
    }

    const gfcLastW = dl.variable(
      dl.randomNormal(
        [this.numGeneratorNeurons, 2], 0,
        1.0 / Math.sqrt(this.numGeneratorNeurons)));
    const gfcLastB = dl.variable(dl.zeros([2]));

    this.gVariables.push(gfcLastW);
    this.gVariables.push(gfcLastB);

    // Discriminator.
    const dfc0W = dl.variable(
      dl.randomNormal(
        [2, this.numDiscriminatorNeurons], 0, 1.0 / Math.sqrt(2)),
      true);
    const dfc0B = dl.variable(dl.zeros([this.numDiscriminatorNeurons]));

    this.dVariables.push(dfc0W);
    this.dVariables.push(dfc0B);

    for (let i = 0; i < this.numDiscriminatorLayers; ++i) {
      const dfcW = dl.variable(
        dl.randomNormal(
          [this.numDiscriminatorNeurons, this.numDiscriminatorNeurons], 0,
          1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
      const dfcB = dl.variable(dl.zeros([this.numDiscriminatorNeurons]));

      this.dVariables.push(dfcW);
      this.dVariables.push(dfcB);
    }

    const dfcLastW = dl.variable(
      dl.randomNormal(
        [this.numDiscriminatorNeurons, 1], 0,
        1.0 / Math.sqrt(this.numDiscriminatorNeurons)));
    const dfcLastB = dl.variable(dl.zeros([1]));

    this.dVariables.push(dfcLastW);
    this.dVariables.push(dfcLastB);

    // Hack to prevent error when using grads (doesn't allow this in model).
    dVariables = this.dVariables;
    numDiscriminatorLayers = this.numDiscriminatorLayers;
    batchSize = this.batchSize;
  }

  generator(noiseTensor: dl.Tensor2D): dl.Tensor2D {
    const gfc0W = this.gVariables[0] as dl.Tensor2D;
    const gfc0B = this.gVariables[1];

    let network = noiseTensor.matMul(gfc0W)
      .add(gfc0B)
      .relu();

    for (let i = 0; i < this.numGeneratorLayers; ++i) {
      const gfcW = this.gVariables[2 + i * 2] as dl.Tensor2D;
      const gfcB = this.gVariables[3 + i * 2];

      network = network.matMul(gfcW)
        .add(gfcB)
        .relu();
    }

    const gfcLastW =
      this.gVariables[2 + this.numGeneratorLayers * 2] as dl.Tensor2D;
    const gfcLastB =
      this.gVariables[3 + this.numGeneratorLayers * 2];

    const generatedTensor: dl.Tensor2D = network.matMul(gfcLastW)
      .add(gfcLastB)
      .tanh() as dl.Tensor2D;

    return generatedTensor;
  }

  discriminator(inputTensor: dl.Tensor2D): dl.Tensor1D {
    const dfc0W = /*this.*/dVariables[0] as dl.Tensor2D;
    const dfc0B = /*this.*/dVariables[1];

    let network = inputTensor.matMul(dfc0W)
      .add(dfc0B)
      .relu();

    for (let i = 0; i < /*this.*/numDiscriminatorLayers; ++i) {
      const dfcW = /*this.*/dVariables[2 + i * 2] as dl.Tensor2D;
      const dfcB = /*this.*/dVariables[3 + i * 2];

      network = network.matMul(dfcW)
        .add(dfcB)
        .relu();
    }
    const dfcLastW =
      /*this.*/dVariables[2 + /*this.*/numDiscriminatorLayers * 2] as
      dl.Tensor2D;
    const dfcLastB =
      /*this.*/dVariables[3 + /*this.*/numDiscriminatorLayers * 2];

    const predictionTensor: dl.Tensor1D =
      network.matMul(dfcLastW)
        .add(dfcLastB)
        .sigmoid()
        .reshape([/*this.*/batchSize]);

    return predictionTensor;
  }

  // Define losses.
  dLoss(truePred: dl.Tensor1D, generatedPred: dl.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return dl.add(
        truePred.sub(dl.scalar(1)).square().mean(),
        generatedPred.square().mean()
      ) as dl.Scalar;
    } else {
      return dl.add(
        truePred.log().mul(dl.scalar(0.95)).mean(),
        dl.sub(dl.scalar(1), generatedPred).log().mean()
      ).mul(dl.scalar(-1)) as dl.Scalar;
    }
  }

  gLoss(generatedPred: dl.Tensor1D) {
    if (this.lossType === 'LeastSq loss') {
      return generatedPred.sub(dl.scalar(1)).square().mean() as dl.Scalar;
    } else {
      return generatedPred.log().mean().mul(dl.scalar(-1)) as dl.Scalar;
    }
  }

  updateOptimizer(
    dOrG: string, optimizerType: string, learningRate: number) {
    if (optimizerType === 'Adam') {
      const beta1 = 0.9;
      const beta2 = 0.999;
      if (dOrG === 'D') {
        this.dOptimizer = dl.train.adam(learningRate, beta1, beta2);
      }
      if (dOrG === 'G') {
        this.gOptimizer = dl.train.adam(learningRate, beta1, beta2);
      }
    } else {
      if (dOrG === 'D') {
        this.dOptimizer = dl.train.sgd(learningRate);
      }
      if (dOrG === 'G') {
        this.gOptimizer = dl.train.sgd(learningRate);
      }
    }
  }
}
