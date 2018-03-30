import * as dl from 'deeplearn';

export abstract class GANLabInputProviderBuilder {
  protected atlas: dl.Tensor2D;
  protected providerCounter: number;

  constructor(protected batchSize: number) {
    this.providerCounter = -1;
  }

  protected abstract generateAtlas(): void;

  abstract getInputProvider(fixStarting?: boolean): dl.InputProvider;
}

export class GANLabNoiseProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(
    private noiseSize: number, private noiseType: string,
    private atlasSize: number, batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    if (this.noiseType === '1D Gaussian' ||
      this.noiseType === '2D Gaussian') {
      this.atlas = dl.truncatedNormal(
        [this.atlasSize, this.noiseSize], 0.5, 0.25);
    } else {
      this.atlas = dl.randomUniform(
        [this.atlasSize, this.noiseSize], 0.0, 1.0);
    }
  }

  getInputProvider(fixStarting?: boolean): dl.InputProvider {
    const provider = this;
    return {
      getNextCopy(): dl.Tensor2D {
        provider.providerCounter++;
        return provider.atlas.slice(
          [fixStarting ? 0 :
            (provider.providerCounter * provider.batchSize) %
            provider.atlasSize, 0],
          [provider.batchSize, provider.noiseSize]
        );
      },
      disposeCopy(copy: dl.Tensor) {
        copy.dispose();
      }
    };
  }

  getNoiseSample(): Float32Array {
    return this.atlas.slice(
      [0, 0], [this.batchSize, this.noiseSize]).dataSync() as Float32Array;
  }
}

export class GANLabTrueSampleProviderBuilder extends
  GANLabInputProviderBuilder {

  private inputAtlasList: number[];

  constructor(
    private atlasSize: number,
    private selectedShapeName: string,
    private drawingPositions: Array<[number, number]>,
    private sampleFromTrueDistribution: Function, batchSize: number) {
    super(batchSize);
    this.inputAtlasList = [];
  }

  generateAtlas() {
    for (let i = 0; i < this.atlasSize; ++i) {
      const distribution = this.sampleFromTrueDistribution(
        this.selectedShapeName, this.drawingPositions);
      this.inputAtlasList.push(distribution[0]);
      this.inputAtlasList.push(distribution[1]);
    }
    this.atlas = dl.Tensor2D.new([this.atlasSize, 2], this.inputAtlasList);
  }

  getInputProvider(fixStarting?: boolean): dl.InputProvider {
    const provider = this;
    return {
      getNextCopy(): dl.Tensor2D {
        provider.providerCounter++;
        return provider.atlas.slice(
          [fixStarting ? 0 :
            (provider.providerCounter * provider.batchSize) %
            provider.atlasSize, 0],
          [provider.batchSize, 2]
        );
      },
      disposeCopy(copy: dl.Tensor) {
        copy.dispose();
      }
    };
  }

  getInputAtlas(): number[] {
    return this.inputAtlasList;
  }
}

export class GANLabUniformNoiseProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(
    private noiseSize: number,
    private numManifoldCells: number, batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    const inputAtlasList = [];
    if (this.noiseSize === 1) {
      for (let i = 0; i < this.numManifoldCells + 1; ++i) {
        inputAtlasList.push(i / this.numManifoldCells);
      }
    } else if (this.noiseSize === 2) {
      for (let i = 0; i < this.numManifoldCells + 1; ++i) {
        for (let j = 0; j < this.numManifoldCells + 1; ++j) {
          inputAtlasList.push(i / this.numManifoldCells);
          inputAtlasList.push(j / this.numManifoldCells);
        }
      }
    }
    while ((inputAtlasList.length / this.noiseSize) % this.batchSize > 0) {
      inputAtlasList.push(0.5);
    }
    this.atlas = dl.Tensor2D.new(
      [inputAtlasList.length / this.noiseSize, this.noiseSize],
      inputAtlasList);
  }

  getInputProvider(): dl.InputProvider {
    const provider = this;
    return {
      getNextCopy(): dl.Tensor2D {
        provider.providerCounter++;
        if (provider.providerCounter * provider.batchSize >
          Math.pow(provider.numManifoldCells + 1, provider.noiseSize)) {
          provider.providerCounter = 0;
        }
        return provider.atlas.slice(
          [
            (provider.providerCounter * provider.batchSize) %
            Math.pow(provider.numManifoldCells + 1, provider.noiseSize),
            0
          ],
          [provider.batchSize, provider.noiseSize]);
      },
      disposeCopy(copy: dl.Tensor) {
        copy.dispose();
      }
    };
  }

  calculateDensitiesForGaussian(): number[] {
    if (this.noiseSize === 2) {
      const densities: number[] = [];
      for (let i = 0; i < this.numManifoldCells; ++i) {
        for (let j = 0; j < this.numManifoldCells; ++j) {
          densities.push(this.probDensity(
            (i + 0.5) / this.numManifoldCells,
            (j + 0.5) / this.numManifoldCells));
        }
      }
      return densities;
    } else {
      return [];
    }
  }

  private probDensity(x: number, y: number) {
    const mu = 0.5;
    const std = 0.25;
    return 1.0 / (2.0 * Math.PI * std * std) * Math.exp(-0.5 /
      (std * std) * ((x - mu) * (x - mu) + (y - mu) * (y - mu)));
  }
}

export class GANLabUniformSampleProviderBuilder extends
  GANLabInputProviderBuilder {

  constructor(private numGridCells: number, batchSize: number) {
    super(batchSize);
  }

  generateAtlas() {
    const inputAtlasList = [];
    for (let j = 0; j < this.numGridCells; ++j) {
      for (let i = 0; i < this.numGridCells; ++i) {
        inputAtlasList.push((i + 0.5) / this.numGridCells);
        inputAtlasList.push((j + 0.5) / this.numGridCells);
      }
    }
    this.atlas = dl.Tensor2D.new(
      [this.numGridCells * this.numGridCells, 2], inputAtlasList);
  }

  getInputProvider(): dl.InputProvider {
    const provider = this;
    return {
      getNextCopy(): dl.Tensor2D {
        provider.providerCounter++;
        return provider.atlas.slice(
          [
            (provider.providerCounter * provider.batchSize) %
            (provider.numGridCells * provider.numGridCells),
            0
          ],
          [provider.batchSize, 2]);
      },
      disposeCopy(copy: dl.Tensor) {
        copy.dispose();
      }
    };
  }
}

export function randNormal() {
  const u = 1 - Math.random();
  const v = 1 - Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
