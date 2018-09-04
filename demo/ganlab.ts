import * as d3 from 'd3-selection';
import { contourDensity } from 'd3-contour';
import { geoPath } from 'd3-geo';
import { scaleSequential } from 'd3-scale';
import { interpolateGreens, interpolatePRGn } from 'd3-scale-chromatic';
import { line } from 'd3-shape';
import * as d3Transition from 'd3-transition';

import { PolymerElement, PolymerHTMLElement } from '../lib/polymer-spec';
import * as tf from '@tensorflow/tfjs-core';

import * as ganlab_input_providers from './ganlab_input_providers';
import * as ganlab_drawing from './ganlab_drawing';
import * as ganlab_evaluators from './ganlab_evaluators';
import * as ganlab_models from './ganlab_models';

const BATCH_SIZE = 150;
const ATLAS_SIZE = 12000;

const NUM_GRID_CELLS = 30;
const NUM_MANIFOLD_CELLS = 20;
const GRAD_ARROW_UNIT_LEN = 0.15;
const NUM_TRUE_SAMPLES_VISUALIZED = 450;

const VIS_INTERVAL = 50;
const EPOCH_INTERVAL = 2;
const SLOW_INTERVAL_MS = 1250;

interface ManifoldCell {
  points: Float32Array[];
  area?: number;
}

// tslint:disable-next-line:variable-name
const GANLabPolymer: new () => PolymerHTMLElement = PolymerElement({
  is: 'gan-lab',
  properties: {
    dLearningRate: Number,
    gLearningRate: Number,
    learningRateOptions: Array,
    dOptimizerType: String,
    gOptimizerType: String,
    optimizerTypeOptions: Array,
    lossType: String,
    lossTypeOptions: Array,
    selectedShapeName: String,
    shapeNames: Array,
    selectedNoiseType: String,
    noiseTypes: Array
  }
});

class GANLab extends GANLabPolymer {
  private iterationCount: number;

  private noiseProvider: ganlab_input_providers.InputProvider;
  private trueSampleProvider: ganlab_input_providers.InputProvider;
  private uniformNoiseProvider: ganlab_input_providers.InputProvider;
  private uniformInputProvider: ganlab_input_providers.InputProvider;

  private usePretrained: boolean;

  private model: ganlab_models.GANLabModel;
  private noiseSize: number;
  private numGeneratorLayers: number;
  private numDiscriminatorLayers: number;
  private numGeneratorNeurons: number;
  private numDiscriminatorNeurons: number;
  private kDSteps: number;
  private kGSteps: number;

  private plotSizePx: number;

  private gDotsElementList: string[];
  private highlightedComponents: HTMLDivElement[];
  private highlightedTooltip: HTMLDivElement;

  private evaluator: ganlab_evaluators.GANLabEvaluatorGridDensities;

  private canvas: HTMLCanvasElement;
  private drawing: ganlab_drawing.GANLabDrawing;

  ready() {
    // HTML elements.
    const numGeneratorLayersElement =
      document.getElementById('num-g-layers') as HTMLElement;
    this.numGeneratorLayers = +numGeneratorLayersElement.innerText;
    document.getElementById('g-layers-add-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers < 5) {
          this.numGeneratorLayers += 1;
          numGeneratorLayersElement.innerText =
            this.numGeneratorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });
    document.getElementById('g-layers-remove-button')!.addEventListener(
      'click', () => {
        if (this.numGeneratorLayers > 0) {
          this.numGeneratorLayers -= 1;
          numGeneratorLayersElement.innerText =
            this.numGeneratorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numDiscriminatorLayersElement =
      document.getElementById('num-d-layers') as HTMLElement;
    this.numDiscriminatorLayers = +numDiscriminatorLayersElement.innerText;
    document.getElementById('d-layers-add-button')!.addEventListener(
      'click', () => {
        if (this.numDiscriminatorLayers < 5) {
          this.numDiscriminatorLayers += 1;
          numDiscriminatorLayersElement.innerText =
            this.numDiscriminatorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });
    document.getElementById('d-layers-remove-button')!.addEventListener(
      'click', () => {
        if (this.numDiscriminatorLayers > 0) {
          this.numDiscriminatorLayers -= 1;
          numDiscriminatorLayersElement.innerText =
            this.numDiscriminatorLayers.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numGeneratorNeuronsElement =
      document.getElementById('num-g-neurons') as HTMLElement;
    this.numGeneratorNeurons = +numGeneratorNeuronsElement.innerText;
    document.getElementById('g-neurons-add-button').addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons < 100) {
          this.numGeneratorNeurons += 1;
          numGeneratorNeuronsElement.innerText =
            this.numGeneratorNeurons.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });
    document.getElementById('g-neurons-remove-button').addEventListener(
      'click', () => {
        if (this.numGeneratorNeurons > 0) {
          this.numGeneratorNeurons -= 1;
          numGeneratorNeuronsElement.innerText =
            this.numGeneratorNeurons.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numDiscriminatorNeuronsElement =
      document.getElementById('num-d-neurons') as HTMLElement;
    this.numDiscriminatorNeurons = +numDiscriminatorNeuronsElement.innerText;
    document.getElementById('d-neurons-add-button').addEventListener(
      'click', () => {
        if (this.numDiscriminatorNeurons < 100) {
          this.numDiscriminatorNeurons += 1;
          numDiscriminatorNeuronsElement.innerText =
            this.numDiscriminatorNeurons.toString();
            this.disabledPretrainedMode();
            this.createExperiment();
        }
      });
    document.getElementById('d-neurons-remove-button').addEventListener(
      'click', () => {
        if (this.numDiscriminatorNeurons > 0) {
          this.numDiscriminatorNeurons -= 1;
          numDiscriminatorNeuronsElement.innerText =
            this.numDiscriminatorNeurons.toString();
          this.disabledPretrainedMode();
          this.createExperiment();
        }
      });

    const numKDStepsElement =
      document.getElementById('k-d-steps') as HTMLElement;
    this.kDSteps = +numKDStepsElement.innerText;
    document.getElementById('k-d-steps-add-button')!.addEventListener(
      'click', () => {
        if (this.kDSteps < 10) {
          this.kDSteps += 1;
          numKDStepsElement.innerText = this.kDSteps.toString();
        }
      });
    document.getElementById('k-d-steps-remove-button')!.addEventListener(
      'click', () => {
        if (this.kDSteps > 0) {
          this.kDSteps -= 1;
          numKDStepsElement.innerText = this.kDSteps.toString();
        }
      });

    const numKGStepsElement =
      document.getElementById('k-g-steps') as HTMLElement;
    this.kGSteps = +numKGStepsElement.innerText;
    document.getElementById('k-g-steps-add-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps < 10) {
          this.kGSteps += 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });
    document.getElementById('k-g-steps-remove-button')!.addEventListener(
      'click', () => {
        if (this.kGSteps > 0) {
          this.kGSteps -= 1;
          numKGStepsElement.innerText = this.kGSteps.toString();
        }
      });

    this.lossTypeOptions = ['Log loss', 'LeastSq loss'];
    this.lossType = 'Log loss';
    this.querySelector('#loss-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.lossType = event.detail.selected;
        this.model.lossType = this.lossType;
      });

    this.learningRateOptions = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];
    this.dLearningRate = 0.1;
    this.querySelector('#d-learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.dLearningRate = +event.detail.selected;
        this.model.updateOptimizer(
          'D', this.dOptimizerType, this.dLearningRate);
      });
    this.gLearningRate = 0.1;
    this.querySelector('#g-learning-rate-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.gLearningRate = +event.detail.selected;
        this.model.updateOptimizer(
          'G', this.gOptimizerType, this.gLearningRate);
      });

    this.optimizerTypeOptions = ['SGD', 'Adam'];
    this.dOptimizerType = 'SGD';
    this.querySelector('#d-optimizer-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.dOptimizerType = event.detail.selected;
        this.model.updateOptimizer(
          'D', this.dOptimizerType, this.dLearningRate);
      });
    this.gOptimizerType = 'SGD';
    this.querySelector('#g-optimizer-type-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.gOptimizerType = event.detail.selected;
        this.model.updateOptimizer(
          'G', this.gOptimizerType, this.gLearningRate);
      });

    this.shapeNames = ['line', 'gaussians', 'ring', 'disjoint', 'drawing'];
    this.selectedShapeName = 'gaussians';

    const distributionElementList = 
      document.querySelectorAll('.distribution-item');

    for (let i = 0; i < distributionElementList.length; ++i) { 
      // tslint:disable-next-line:no-any event has no type
      distributionElementList[i].addEventListener('click', (event: any) =>
        this.changeDataset(event.target), false);
    }  

    this.noiseTypes =
      ['1D Uniform', '1D Gaussian', '2D Uniform', '2D Gaussian'];
    this.selectedNoiseType = '2D Uniform';
    this.noiseSize = 2;
    this.querySelector('#noise-dropdown')!.addEventListener(
      // tslint:disable-next-line:no-any event has no type
      'iron-activate', (event: any) => {
        this.selectedNoiseType = event.detail.selected;
        this.noiseSize = +this.selectedNoiseType.substring(0, 1);
        this.disabledPretrainedMode();
        this.createExperiment();
      });

    // Checkbox toggles.
    const checkboxList = [
      {
        graph: '#overlap-plots', 
        description: '#toggle-right-discriminator', 
        layer: '#vis-discriminator-output'
      },
      {
        graph: '#enable-manifold', 
        description: '#toggle-right-generator',
        layer: '#vis-manifold'
      },
      {
        graph: '#show-t-samples', 
        description: '#toggle-right-real-samples', 
        layer: '#vis-true-samples'
      },
      {
        graph: '#show-g-samples', 
        description: '#toggle-right-fake-samples', 
        layer: '#vis-generated-samples'
      },
      {
        graph: '#show-g-gradients', 
        description: '#toggle-right-gradients', 
        layer: '#vis-generator-gradients'
      }
    ];
    checkboxList.forEach(layer => {      
      this.querySelector(layer.graph)!.addEventListener(
        'change', (event: Event) => {
        const container =
          this.querySelector(layer.layer) as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).checked ? 'visible' : 'hidden';
          
        const element = 
          this.querySelector(layer.description) as HTMLElement;
        // tslint:disable-next-line:no-any
        if ((event.target as any).checked) {
          element.classList.add('checked');
        } else {
          element.classList.remove('checked');
        }
      });
      this.querySelector(layer.description)!.addEventListener(
        'click', (event: Event) => {
        const spanElement = 
          this.querySelector(layer.description) as HTMLElement;
        const container =
          this.querySelector(layer.layer) as HTMLElement;
        const element = 
          this.querySelector(layer.graph) as HTMLInputElement;

        // tslint:disable-next-line:no-any
        if ((event.target as any).classList.contains('checked')) {
          spanElement.classList.remove('checked');
          container.style.visibility = 'hidden';
          element.checked = false;
        } else {
          spanElement.classList.add('checked');
          container.style.visibility = 'visible'
          element.checked = true;
        }
      });
    });
    this.querySelector('#show-t-contour')!.addEventListener(
      'change', (event: Event) => {
        const container =
          this.querySelector('#vis-true-samples-contour') as SVGGElement;
        // tslint:disable-next-line:no-any
        container.style.visibility =
          (event.target as any).checked ? 'visible' : 'hidden';
      });

    // Pre-trained checkbox.
    this.usePretrained = true;
    this.querySelector('#toggle-pretrained')!.addEventListener(
      'change', (event: Event) => {
        // tslint:disable-next-line:no-any
        this.usePretrained = (event.target as any).checked;
        this.loadModelAndCreateExperiment();
      });
      
    // Timeline controls.
    document.getElementById('play-pause-button').addEventListener(
      'click', () => this.onClickPlayPauseButton());
    document.getElementById('reset-button').addEventListener(
      'click', () => this.onClickResetButton());

    document.getElementById('next-step-d-button').addEventListener(
      'click', () => this.onClickNextStepButton('D'));
    document.getElementById('next-step-g-button').addEventListener(
      'click', () => this.onClickNextStepButton('G'));
    document.getElementById('next-step-all-button').addEventListener(
      'click', () => this.onClickNextStepButton());

    this.stepMode = false;
    document.getElementById('next-step-button').addEventListener(
      'click', () => this.onClickStepModeButton());

    this.slowMode = false;
    document.getElementById('slow-step')!.addEventListener(
      'click', () => this.onClickSlowModeButton());

    this.editMode = true;
    document.getElementById('edit-model-button')!.addEventListener(
      'click', () => this.onClickEditModeButton());
    this.onClickEditModeButton();

    this.iterCountElement =
      document.getElementById('iteration-count') as HTMLElement;

    document.getElementById('save-model')!.addEventListener(
        'click', () => this.onClickSaveModelButton());
  
    // Visualization.
    this.plotSizePx = 400;
    this.mediumPlotSizePx = 140;
    this.smallPlotSizePx = 50;

    this.colorScale = interpolatePRGn;

    this.gDotsElementList = [
      '#vis-generated-samples',
      '#svg-generated-samples',
      '#svg-generated-prediction'
    ];
    this.dFlowElements =
      this.querySelectorAll('.d-update-flow') as NodeListOf<SVGPathElement>;
    this.gFlowElements =
      this.querySelectorAll('.g-update-flow') as NodeListOf<SVGPathElement>;

    // Generator animation.
    document.getElementById('svg-generator-manifold')!.addEventListener(
      'mouseenter', () => {
        this.playGeneratorAnimation();
      });

    // Drawing-related.
    this.canvas =
      document.getElementById('input-drawing-canvas') as HTMLCanvasElement;
    this.drawing = new ganlab_drawing.GANLabDrawing(
      this.canvas, this.plotSizePx);

    this.finishDrawingButton =
      document.getElementById('finish-drawing') as HTMLInputElement;
    this.finishDrawingButton.addEventListener(
      'click', () => this.onClickFinishDrawingButton());

    // Create a new experiment.
    this.loadModelAndCreateExperiment();
  }

  private createExperiment() {
    // Reset.
    this.pause();
    this.iterationCount = 0;
    this.iterCountElement.innerText = this.zeroPad(this.iterationCount);

    this.isPausedOngoingIteration = false;

    document.getElementById('d-loss-value').innerText = '-';
    document.getElementById('g-loss-value').innerText = '-';
    document.getElementById('d-loss-bar').style.width = '0';
    document.getElementById('g-loss-bar').style.width = '0';
    this.recreateCharts();

    const dataElements = [
      d3.select('#vis-true-samples').selectAll('.true-dot'),
      d3.select('#svg-true-samples').selectAll('.true-dot'),
      d3.select('#svg-true-prediction').selectAll('.true-dot'),
      d3.select('#vis-true-samples-contour').selectAll('path'),
      d3.select('#svg-noise').selectAll('.noise-dot'),
      d3.select('#vis-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-samples').selectAll('.generated-dot'),
      d3.select('#svg-generated-prediction').selectAll('.generated-dot'),
      d3.select('#vis-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#svg-discriminator-output').selectAll('.uniform-dot'),
      d3.select('#vis-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#vis-manifold').selectAll('.manifold-cells'),
      d3.select('#vis-manifold').selectAll('.grids'),
      d3.select('#svg-generator-manifold').selectAll('.uniform-generated-dot'),
      d3.select('#svg-generator-manifold').selectAll('.manifold-cells'),
      d3.select('#svg-generator-manifold').selectAll('.grids'),
      d3.select('#vis-generator-gradients').selectAll('.gradient-generated'),
      d3.select('#svg-generator-gradients').selectAll('.gradient-generated')
    ];
    dataElements.forEach((element) => {
      element.data([]).exit().remove();
    });

    // Input providers.
    const noiseProviderBuilder =
      new ganlab_input_providers.GANLabNoiseProviderBuilder(
        this.noiseSize, this.selectedNoiseType,
        ATLAS_SIZE, BATCH_SIZE);
    noiseProviderBuilder.generateAtlas();
    this.noiseProvider = noiseProviderBuilder.getInputProvider();
    this.noiseProviderFixed = noiseProviderBuilder.getInputProvider(true);

    const drawingPositions = this.drawing.drawingPositions;
    const trueSampleProviderBuilder =
      new ganlab_input_providers.GANLabTrueSampleProviderBuilder(
        ATLAS_SIZE, this.selectedShapeName,
        drawingPositions, BATCH_SIZE);
    trueSampleProviderBuilder.generateAtlas();
    this.trueSampleProvider = trueSampleProviderBuilder.getInputProvider();
    this.trueSampleProviderFixed =
      trueSampleProviderBuilder.getInputProvider(true);

    if (this.noiseSize <= 2) {
      const uniformNoiseProviderBuilder =
        new ganlab_input_providers.GANLabUniformNoiseProviderBuilder(
          this.noiseSize, NUM_MANIFOLD_CELLS, BATCH_SIZE);
      uniformNoiseProviderBuilder.generateAtlas();
      if (this.selectedNoiseType === '2D Gaussian') {
        this.densitiesForGaussian =
          uniformNoiseProviderBuilder.calculateDensitiesForGaussian();
      }
      this.uniformNoiseProvider =
        uniformNoiseProviderBuilder.getInputProvider();
    }

    const uniformSampleProviderBuilder =
      new ganlab_input_providers.GANLabUniformSampleProviderBuilder(
        NUM_GRID_CELLS, BATCH_SIZE);
    uniformSampleProviderBuilder.generateAtlas();
    this.uniformInputProvider = uniformSampleProviderBuilder.getInputProvider();

    // Visualize true samples.
    this.visualizeTrueDistribution(trueSampleProviderBuilder.getInputAtlas());

    // Visualize noise samples.
    this.visualizeNoiseDistribution(noiseProviderBuilder.getNoiseSample());

    // Initialize evaluator.
    this.evaluator =
      new ganlab_evaluators.GANLabEvaluatorGridDensities(NUM_GRID_CELLS);
    this.evaluator.createGridsForTrue(
      trueSampleProviderBuilder.getInputAtlas(), NUM_TRUE_SAMPLES_VISUALIZED);

    // Prepare for model.
    this.model = new ganlab_models.GANLabModel(
      this.noiseSize, this.numGeneratorLayers, this.numDiscriminatorLayers,
      this.numGeneratorNeurons, this.numDiscriminatorNeurons,
      BATCH_SIZE, this.lossType);
    this.model.initializeModelVariables();
    this.model.updateOptimizer('D', this.dOptimizerType, this.dLearningRate);
    this.model.updateOptimizer('G', this.gOptimizerType, this.gLearningRate);
  }

  private changeDataset(element: HTMLElement) {
    this.selectedShapeName = element.getAttribute('data-distribution-name');

    const distributionElementList = 
      document.querySelectorAll('.distribution-item');
    for (let i = 0; i < distributionElementList.length; ++i) {
      if (distributionElementList[i].classList.contains('selected')) {
        distributionElementList[i].classList.remove('selected');
      }
    }
    if (!element.classList.contains('selected')) {
      element.classList.add('selected');
    }

    this.disabledPretrainedMode();
    this.loadModelAndCreateExperiment();
  }
    
  private loadModelAndCreateExperiment() {
    if (this.selectedShapeName === 'drawing') {
      this.pause();
      this.drawing.prepareDrawing();
      this.disabledPretrainedMode();
    } else if (this.usePretrained === true) {
      const filename = `pretrained_${this.selectedShapeName}`;
      this.loadPretrainedWeightFile(filename).then((loadedModel) => {
        const loadedIterCount = this.iterationCount;

        this.createExperiment();
        this.model.loadPretrainedWeights(loadedModel);

        // Run one iteration for visualization.
        this.isPlaying = true;
        this.iterateTraining(false);
        this.isPlaying = false;

        this.iterationCount = loadedIterCount;
        this.iterCountElement.innerText = this.zeroPad(this.iterationCount);
      });
    } else {
      const filename = `pretrained_${this.selectedShapeName}`;
      this.loadPretrainedWeightFile(filename).then((loadedModel) => {
        this.createExperiment();
      });
    }
  }

  private visualizeTrueDistribution(inputAtlasList: number[]) {
    const color = scaleSequential(interpolateGreens)
      .domain([0, 0.05]);

    const trueDistribution: Array<[number, number]> = [];
    while (trueDistribution.length < NUM_TRUE_SAMPLES_VISUALIZED) {
      const values = inputAtlasList.splice(0, 2);
      trueDistribution.push([values[0], values[1]]);
    }

    const contour = contourDensity()
      .x((d: number[]) => d[0] * this.plotSizePx)
      .y((d: number[]) => (1.0 - d[1]) * this.plotSizePx)
      .size([this.plotSizePx, this.plotSizePx])
      .bandwidth(15)
      .thresholds(5);

    d3.select('#vis-true-samples-contour')
      .selectAll('path')
      .data(contour(trueDistribution))
      .enter()
      .append('path')
      .attr('fill', (d: any) => color(d.value))
      .attr('data-value', (d: any) => d.value)
      .attr('d', geoPath());

    const trueDotsElementList = [
      '#vis-true-samples',
      '#svg-true-samples',
    ];
    trueDotsElementList.forEach((dotsElement, k) => {
      const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
      const radius = k === 0 ? 2 : 1;
      d3.select(dotsElement)
        .selectAll('.true-dot')
        .data(trueDistribution)
        .enter()
        .append('circle')
        .attr('class', 'true-dot gan-lab')
        .attr('r', radius)
        .attr('cx', (d: number[]) => d[0] * plotSizePx)
        .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx)
        .append('title')
        .text((d: number[]) => `${d[0].toFixed(2)}, ${d[1].toFixed(2)}`);
    });
  }

  private visualizeNoiseDistribution(inputList: Float32Array) {
    const noiseSamples: number[][] = [];
    for (let i = 0; i < inputList.length / this.noiseSize; ++i) {
      const values = [];
      for (let j = 0; j < this.noiseSize; ++j) {
        values.push(inputList[i * this.noiseSize + j]);
      }
      noiseSamples.push(values);
    }

    d3.select('#svg-noise')
      .selectAll('.noise-dot')
      .data(noiseSamples)
      .enter()
      .append('circle')
      .attr('class', 'noise-dot gan-lab')
      .attr('r', 1)
      .attr('cx', (d: number[]) => d[0] * this.smallPlotSizePx)
      .attr('cy', (d: number[]) => this.noiseSize === 1
        ? this.smallPlotSizePx / 2
        : (1.0 - d[1]) * this.smallPlotSizePx)
      .append('title')
      .text((d: number[], i: number) => this.noiseSize === 1
        ? `${Number(d[0]).toFixed(2)} (${i})`
        : `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)} (${i})`);
  }

  private onClickFinishDrawingButton() {
    if (this.drawing.drawingPositions.length === 0) {
      alert('Draw something on canvas');
    } else {
      const drawingElement =
        this.querySelector('#drawing-container') as HTMLElement;
      drawingElement.style.display = 'none';
      const drawingBackgroundElement =
        this.querySelector('#drawing-disable-background') as HTMLDivElement;
      drawingBackgroundElement.style.display = 'none';
      this.createExperiment();
    }
  }

  private disabledPretrainedMode() {
    this.usePretrained = false;
    const element = 
      document.getElementById('toggle-pretrained') as HTMLInputElement;
    element.checked = false;
  }

  private play() {
    if (this.stepMode) {
      this.onClickStepModeButton();
    }

    this.isPlaying = true;
    document.getElementById('play-pause-button')!.classList.add('playing');
    if (!this.isPausedOngoingIteration) {
      this.iterateTraining(true);
    }
    document.getElementById('model-vis-svg').classList.add('playing');
  }

  private pause() {
    // Extra iteration for visualization.
    this.iterateTraining(false);
    this.isPlaying = false;
    const button = document.getElementById('play-pause-button');
    if (button.classList.contains('playing')) {
      button.classList.remove('playing');
    }
    document.getElementById('model-vis-svg').classList.remove('playing');
  }

  private onClickPlayPauseButton() {
    if (this.isPlaying) {
      this.pause();
    } else {
      this.play();
    }
  }

  private onClickNextStepButton(type?: string) {
    if (this.isPlaying) {
      this.pause();
    }
    this.isPlaying = true;
    this.iterateTraining(false, type);
    this.isPlaying = false;
  }

  private onClickResetButton() {
    if (this.isPlaying) {
      this.pause();
    }
    this.loadModelAndCreateExperiment();
  }

  private onClickStepModeButton() {
    if (!this.stepMode) {
      if (this.isPlaying) {
        this.pause();
      }
      if (this.slowMode) {
        this.onClickSlowModeButton();
      }

      this.stepMode = true;
      document.getElementById('next-step-button')
        .classList.add('mdl-button--colored');
      document.getElementById('step-buttons').style.display = 'block';
    } else {
      this.stepMode = false;
      document.getElementById('next-step-button')
        .classList.remove('mdl-button--colored');
      document.getElementById('step-buttons').style.display = 'none';
    }
  }

  private onClickSlowModeButton() {
    if (this.editMode) {
      this.onClickEditModeButton();
    }
    this.slowMode = !this.slowMode;

    if (this.slowMode === true) {
      if (this.stepMode) {
        this.onClickStepModeButton();
      }
      document.getElementById('slow-step')
        .classList.add('mdl-button--colored');
      document.getElementById('tooltips').classList.add('shown');
    } else {
      document.getElementById('slow-step')
        .classList.remove('mdl-button--colored');
      this.dehighlightStep();
      const container =
        document.getElementById('model-visualization-container');
      if (container.classList.contains('any-highlighted')) {
        container.classList.remove('any-highlighted');
      }
      document.getElementById(
        'component-generator').classList.remove('deactivated');
      document.getElementById(
        'component-discriminator').classList.remove('deactivated');
      document.getElementById(
        'component-d-loss').classList.remove('activated');
      document.getElementById(
        'component-g-loss').classList.remove('activated');
      for (let i = 0; i < this.dFlowElements.length; ++i) {
        this.dFlowElements[i].classList.remove('d-activated');
      }
      for (let i = 0; i < this.gFlowElements.length; ++i) {
        this.gFlowElements[i].classList.remove('g-activated');
      }
      document.getElementById('tooltips')!.classList.remove('shown');
    }
  }

  private onClickEditModeButton() {
    const elements: NodeListOf<HTMLDivElement> =
      this.querySelectorAll('.config-item');
    for (let i = 0; i < elements.length; ++i) {
      elements[i].style.visibility =
        this.editMode ? 'hidden' : 'visible';
    }
    this.editMode = !this.editMode;
    if (this.editMode === true) {
      document.getElementById('edit-model-button')
        .classList.add('mdl-button--colored');
    } else {
      document.getElementById('edit-model-button')
        .classList.remove('mdl-button--colored');
    }
  }

  private zeroPad(n: number): string {
    const pad = '000000';
    return (pad + n).slice(-pad.length).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  private async iterateTraining(keepIterating: boolean, type?: string) {
    if (!this.isPlaying) {
      return;
    }

    this.iterationCount++;

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
      this.iterationCount % EPOCH_INTERVAL === 0) {
      this.iterCountElement.innerText = this.zeroPad(this.iterationCount);

      d3.select('#model-vis-svg')
        .selectAll('path')
        .style('stroke-dashoffset', () => this.iterationCount * (-1));
    }

    // Visualize generated samples before training.
    if (this.slowMode) {
      const container =
        document.getElementById('model-visualization-container');
      if (!container.classList.contains('any-highlighted')) {
        container.classList.add('any-highlighted');
      }
      document.getElementById(
        'component-generator').classList.add('deactivated');
      document.getElementById(
        'component-d-loss').classList.add('activated');
      for (let i = 0; i < this.dFlowElements.length; ++i) {
        this.dFlowElements[i].classList.add('d-activated');
      }
      await this.sleep(SLOW_INTERVAL_MS);

      await this.highlightStep(true,
        ['component-noise', 'component-generator',
          'component-generated-samples'],
        'tooltip-d-generated-samples');
    }

    tf.tidy(() => {
      let gResultData: Float32Array;
      if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % VIS_INTERVAL === 0) {
        const gDataBefore: Array<[number, number]> = [];
        const noiseFixedBatch =
          this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gResult = this.model.generator(noiseFixedBatch);
        gResultData = gResult.dataSync() as Float32Array;
        for (let j = 0; j < gResultData.length / 2; ++j) {
          gDataBefore.push([gResultData[j * 2], gResultData[j * 2 + 1]]);
        }

        if (this.iterationCount === 1) {
          this.gDotsElementList.forEach((dotsElement, k) => {
            const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
            const radius = k === 0 ? 2 : 1;
            d3.select(dotsElement).selectAll('.generated-dot')
              .data(gDataBefore)
              .enter()
              .append('circle')
              .attr('class', 'generated-dot gan-lab')
              .attr('r', radius)
              .attr('cx', (d: number[]) => d[0] * plotSizePx)
              .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx)
              .append('title')
              .text((d: number[]) =>
                `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)}`);
          });
        } else {
          this.gDotsElementList.forEach((dotsElement, k) => {
            const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
            d3Transition.transition()
              .select(dotsElement)
              .selectAll('.generated-dot')
              .selection().data(gDataBefore)
              .transition().duration(SLOW_INTERVAL_MS / 600)
              .attr('cx', (d: number[]) => d[0] * plotSizePx)
              .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx);
          });
        }
      }
    });

    if (this.slowMode) {
      await this.highlightStep(true,
        ['component-true-samples', 'component-generated-samples',
          'component-discriminator',
          'component-true-prediction', 'component-generated-prediction'],
        'tooltip-d-prediction');
    }

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
      this.iterationCount % VIS_INTERVAL === 0) {
      tf.tidy(() => {
        const noiseBatch =
          this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const trueSampleBatch =
          this.trueSampleProviderFixed.getNextCopy() as tf.Tensor2D;
        const truePred = this.model.discriminator(trueSampleBatch);
        const generatedPred =
          this.model.discriminator(this.model.generator(noiseBatch));

        const inputData1 = trueSampleBatch.dataSync();
        const resultData1 = truePred.dataSync();
        const resultData2 = generatedPred.dataSync();
        const pInputData1: number[][] = [];
        const pData1: number[] = [];
        const pData2: number[] = [];
        for (let i = 0; i < inputData1.length / 2; ++i) {
          pInputData1.push([inputData1[i * 2], inputData1[i * 2 + 1]]);
        }
        for (let i = 0; i < resultData1.length; ++i) {
          pData1.push(resultData1[i]);
        }
        for (let i = 0; i < resultData2.length; ++i) {
          pData2.push(resultData2[i]);
        }

        if (this.iterationCount === 1) {
          d3.select('#svg-true-prediction')
            .selectAll('.true-dot')
            .data(pInputData1)
            .enter()
            .append('circle')
            .attr('class', 'true-dot gan-lab')
            .attr('r', 1)
            .attr('cx', (d: number[]) => d[0] * this.smallPlotSizePx)
            .attr('cy', (d: number[]) => (1.0 - d[1]) * this.smallPlotSizePx);
        }
        const sqrtAbs = (d: number) => {
          if (d > 0.5) {
            return Math.pow(d * 2.0 - 1.0, 0.5) * 0.5 + 0.5;
          } else if (d < 0.5) {
            return Math.pow((d * 2.0 - 1.0) * (-1), 0.5) * (-0.5) + 0.5;
          } else {
            return 0.5;
          }
        };
        d3.select('#svg-true-prediction')
          .selectAll('.true-dot')
          .data(pData1)
          .style('fill', (d: number) => this.colorScale(sqrtAbs(d)));
        if (this.iterationCount > 1 || this.usePretrained) {
          d3.select('#svg-generated-prediction')
            .selectAll('.generated-dot')
            .data(pData2)
            .style('fill', (d: number) => this.colorScale(sqrtAbs(d)));
        }
      });
    }

    // Train Discriminator.
    let dCostVal: number = null;
    tf.tidy(() => {
      const kDSteps = type === 'D' ? 1 : (type === 'G' ? 0 : this.kDSteps);
      for (let j = 0; j < kDSteps; j++) {
        const dCost = this.model.dOptimizer.minimize(() => {
          const noiseBatch = this.noiseProvider.getNextCopy() as tf.Tensor2D;
          const trueSampleBatch =
            this.trueSampleProvider.getNextCopy() as tf.Tensor2D;
          const truePred = this.model.discriminator(trueSampleBatch);
          const generatedPred =
            this.model.discriminator(this.model.generator(noiseBatch));
          return this.model.dLoss(truePred, generatedPred);
        }, true, this.model.dVariables);
        if ((!keepIterating || this.iterationCount === 1 || this.slowMode ||
          this.iterationCount % VIS_INTERVAL === 0)
          && j + 1 === kDSteps) {
          dCostVal = dCost.get();
        }
      }
    });

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
      this.iterationCount % VIS_INTERVAL === 0) {

      if (this.slowMode) {
        await this.highlightStep(true, ['component-d-loss'], 'tooltip-d-loss');
      }

      // Update discriminator loss.
      if (dCostVal) {
        document.getElementById('d-loss-value').innerText =
          (dCostVal / 2).toFixed(3);
        document.getElementById('d-loss-bar').title = (dCostVal / 2).toFixed(3);
        document.getElementById('d-loss-bar').style.width =
          this.model.lossType === 'LeastSq loss'
            ? `${dCostVal * 50.0}px`
            : `${Math.pow(dCostVal * 0.5, 2) * 50.0}px`;
      }

      if (this.slowMode) {
        await this.highlightStep(true,
          ['component-discriminator-gradients'], 'tooltip-d-gradients');
      }

      if (this.slowMode) {
        await this.highlightStep(true,
          ['component-discriminator'], 'tooltip-update-discriminator');
      }

      // Visualize discriminator's output.
      const dData: number[] = [];
      tf.tidy(() => {
        for (let i = 0; i < NUM_GRID_CELLS * NUM_GRID_CELLS / BATCH_SIZE; ++i) {
          const inputBatch =
            this.uniformInputProvider.getNextCopy() as tf.Tensor2D;
          const result = this.model.discriminator(inputBatch);
          const resultData = result.dataSync();
          for (let j = 0; j < resultData.length; ++j) {
            dData.push(resultData[j]);
          }
        }

        const gridDotsElementList = [
          '#vis-discriminator-output',
          '#svg-discriminator-output'
        ];
        if (this.iterationCount === 1) {
          gridDotsElementList.forEach((dotsElement, k) => {
            const plotSizePx = k === 0 ? this.plotSizePx :
              (k === 1 ? this.mediumPlotSizePx : this.smallPlotSizePx);
            d3.select(dotsElement)
              .selectAll('.uniform-dot')
              .data(dData)
              .enter()
              .append('rect')
              .attr('class', 'uniform-dot gan-lab')
              .attr('width', plotSizePx / NUM_GRID_CELLS)
              .attr('height', plotSizePx / NUM_GRID_CELLS)
              .attr(
                'x',
                (d: number, i: number) =>
                  (i % NUM_GRID_CELLS) * (plotSizePx / NUM_GRID_CELLS))
              .attr(
                'y',
                (d: number, i: number) => plotSizePx -
                  (Math.floor(i / NUM_GRID_CELLS) + 1) *
                  (plotSizePx / NUM_GRID_CELLS))
              .style('fill', (d: number) => this.colorScale(d));
          });
        }
        gridDotsElementList.forEach((dotsElement) => {
          d3.select(dotsElement)
            .selectAll('.uniform-dot')
            .data(dData)
            .style('fill', (d: number) => this.colorScale(d));
        });
      });
    }

    if (this.slowMode) {
      await this.sleep(SLOW_INTERVAL_MS);
      this.dehighlightStep();

      document.getElementById(
        'component-generator').classList.remove('deactivated');
      document.getElementById(
        'component-d-loss').classList.remove('activated');
      for (let i = 0; i < this.dFlowElements.length; ++i) {
        this.dFlowElements[i].classList.remove('d-activated');
      }

      document.getElementById(
        'component-discriminator').classList.add('deactivated');
      document.getElementById(
        'component-g-loss').classList.add('activated');
      for (let i = 0; i < this.gFlowElements.length; ++i) {
        this.gFlowElements[i].classList.add('g-activated');
      }
      await this.sleep(SLOW_INTERVAL_MS);

      await this.highlightStep(false,
        ['component-noise', 'component-generator',
          'component-generated-samples'],
        'tooltip-g-generated-samples');
    }

    if (this.slowMode) {
      await this.highlightStep(false,
        ['component-generated-samples', 'component-discriminator',
          'component-generated-prediction'],
        'tooltip-g-prediction');
    }

    if (this.slowMode) {
      await this.highlightStep(false, ['component-g-loss'], 'tooltip-g-loss');
    }

    // Visualize generated samples before training.
    const gradData: Array<[number, number, number, number]> = [];
    tf.tidy(() => {
      let gResultData: Float32Array;
      if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % VIS_INTERVAL === 0) {
        const gDataBefore: Array<[number, number]> = [];
        const noiseFixedBatch =
          this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gResult = this.model.generator(noiseFixedBatch);
        gResultData = gResult.dataSync() as Float32Array;
        for (let j = 0; j < gResultData.length / 2; ++j) {
          gDataBefore.push([gResultData[j * 2], gResultData[j * 2 + 1]]);
        }

        this.gDotsElementList.forEach((dotsElement, k) => {
          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
          d3Transition.transition()
            .select(dotsElement)
            .selectAll('.generated-dot')
            .selection().data(gDataBefore)
            .transition().duration(SLOW_INTERVAL_MS / 600)
            .attr('cx', (d: number[]) => d[0] * plotSizePx)
            .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx);
        });
      }

      // Compute and store gradients before training.
      if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
        this.iterationCount % VIS_INTERVAL === 0) {
        const gradFunction = tf.grad(this.model.discriminator);
        const noiseFixedBatchForGrad =
          this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gSamples = this.model.generator(noiseFixedBatchForGrad);
        const grad = gradFunction(gSamples);
        const gGradient = grad.dataSync();

        for (let i = 0; i < gResultData.length / 2; ++i) {
          gradData.push([
            gResultData[i * 2], gResultData[i * 2 + 1],
            gGradient[i * 2], gGradient[i * 2 + 1]
          ]);
        }
      }
    });

    // Train generator.
    const kGSteps = type === 'G' ? 1 : (type === 'D' ? 0 : this.kGSteps);
    let gCostVal: number = null;
    tf.tidy(() => {
      for (let j = 0; j < kGSteps; j++) {
        const gCost = this.model.gOptimizer.minimize(() => {
          const noiseBatch = this.noiseProvider.getNextCopy() as tf.Tensor2D;
          const pred =
            this.model.discriminator(this.model.generator(noiseBatch));
          return this.model.gLoss(pred);
        }, true, this.model.gVariables);
        if ((!keepIterating || this.iterationCount === 1 || this.slowMode ||
          this.iterationCount % VIS_INTERVAL === 0)
          && j + 1 === kGSteps) {
          gCostVal = gCost.get();
        }
      }
    });

    if (!keepIterating || this.iterationCount === 1 || this.slowMode ||
      this.iterationCount % VIS_INTERVAL === 0) {
      // Update generator loss.
      if (gCostVal) {
        document.getElementById('g-loss-value').innerText =
          gCostVal.toFixed(3);
        document.getElementById('g-loss-bar').title = gCostVal.toFixed(3);
        document.getElementById('g-loss-bar').style.width =
          this.model.lossType === 'LeastSq loss'
            ? `${gCostVal * 2.0 * 50.0}px`
            : `${Math.pow(gCostVal, 2) * 50.0}px`;
      }

      // Update charts.
      if (this.iterationCount === 1) {
        const chartContainer =
          document.getElementById('chart-container') as HTMLElement;
        chartContainer.style.visibility = 'visible';
      }

      this.updateChartData(this.costChartData, this.iterationCount,
        [dCostVal ? dCostVal / 2 : null, gCostVal]);
      this.costChart.update();

      if (this.slowMode) {
        await this.highlightStep(false,
          ['component-generator-gradients'], 'tooltip-g-gradients');
      }

      // Visualize gradients for generator.
      // Values already computed above.
      const gradDotsElementList = [
        '#vis-generator-gradients',
        '#svg-generator-gradients'
      ];
      if (this.iterationCount === 1) {
        gradDotsElementList.forEach((dotsElement, k) => {
          const plotSizePx = k === 0 ?
            this.plotSizePx : this.smallPlotSizePx;
          const arrowWidth = k === 0 ? 0.002 : 0.001;
          d3.select(dotsElement)
            .selectAll('.gradient-generated')
            .data(gradData)
            .enter()
            .append('polygon')
            .attr('class', 'gradient-generated gan-lab')
            .attr('points', (d: number[]) =>
              this.createArrowPolygon(d, plotSizePx, arrowWidth));
        });
      }

      gradDotsElementList.forEach((dotsElement, k) => {
        const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
        const arrowWidth = k === 0 ? 0.002 : 0.001;
        d3Transition.transition()
          .select(dotsElement)
          .selectAll('.gradient-generated').selection().data(gradData)
          .transition().duration(SLOW_INTERVAL_MS)
          .attr('points', (d: number[]) =>
            this.createArrowPolygon(d, plotSizePx, arrowWidth));
      });

      if (this.slowMode) {
        await this.highlightStep(false,
          ['component-generator'], 'tooltip-update-generator');
      }

      // Visualize manifold for 1-D or 2-D noise.
      tf.tidy(() => {
        if (this.noiseSize <= 2) {
          const manifoldData: Float32Array[] = [];
          const numBatches = Math.ceil(Math.pow(
            NUM_MANIFOLD_CELLS + 1, this.noiseSize) / BATCH_SIZE);
          const remainingDummy = BATCH_SIZE * numBatches - Math.pow(
            NUM_MANIFOLD_CELLS + 1, this.noiseSize) * this.noiseSize;
          for (let k = 0; k < numBatches; ++k) {
            const noiseBatch =
              this.uniformNoiseProvider.getNextCopy() as tf.Tensor2D;
            const result = this.model.generator(noiseBatch);
            const maniResult: Float32Array = result.dataSync() as Float32Array;
            for (let i = 0; i < (k + 1 < numBatches ?
              BATCH_SIZE : BATCH_SIZE - remainingDummy); ++i) {
              manifoldData.push(maniResult.slice(i * 2, i * 2 + 2));
            }
          }

          // Create grid cells.
          const gridData: ManifoldCell[] = this.noiseSize === 1
            ? [{ points: manifoldData }]
            : this.createGridCellsFromManifoldData(manifoldData);

          const gManifoldElementList = [
            '#vis-manifold',
            '#svg-generator-manifold'
          ];
          gManifoldElementList.forEach((gManifoldElement, k) => {
            const plotSizePx =
              k === 0 ? this.plotSizePx : this.mediumPlotSizePx;
            const manifoldCell =
              line()
                .x((d: number[]) => d[0] * plotSizePx)
                .y((d: number[]) => (1.0 - d[1]) * plotSizePx);

            if (this.iterationCount === 1) {
              d3.select(gManifoldElement)
                .selectAll('.grids')
                .data(gridData)
                .enter()
                .append('g')
                .attr('class', 'grids gan-lab')
                .append('path')
                .attr('class', 'manifold-cell gan-lab')
                .style('fill', () => {
                  return this.noiseSize === 2 ? '#7b3294' : 'none';
                });
            }
            d3.select(gManifoldElement)
              .selectAll('.grids')
              .data(gridData)
              .select('.manifold-cell')
              .attr('d', (d: ManifoldCell) => manifoldCell(
                d.points.map(point => [point[0], point[1]] as [number, number])
              ))
              .style('fill-opacity', (d: ManifoldCell, i: number) => {
                return this.selectedNoiseType === '2D Gaussian'
                  ? Math.min(0.1 + this.densitiesForGaussian[i] /
                    (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.2, 0.9)
                  : (this.noiseSize === 2 ? Math.max(
                    0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
                    : 'none');
              });

            if (this.noiseSize === 1) {
              const manifoldDots =
                d3.select(gManifoldElement)
                  .selectAll('.uniform-generated-dot')
                  .data(manifoldData);
              if (this.iterationCount === 1) {
                manifoldDots.enter()
                  .append('circle')
                  .attr('class', 'uniform-generated-dot gan-lab')
                  .attr('r', 1);
              }
              manifoldDots
                .attr('cx', (d: Float32Array) => d[0] * plotSizePx)
                .attr('cy', (d: Float32Array) => (1.0 - d[1]) * plotSizePx);
            }
          });
        }
      });

      const gData: Array<[number, number]> = [];
      tf.tidy(() => {
        const noiseFixedBatch =
          this.noiseProviderFixed.getNextCopy() as tf.Tensor2D;
        const gResult = this.model.generator(noiseFixedBatch);
        const gResultData = gResult.dataSync();
        for (let i = 0; i < gResultData.length / 2; ++i) {
          gData.push([gResultData[i * 2], gResultData[i * 2 + 1]]);
        }
      });

      // Visualize generated samples.
      if (!this.slowMode) {
        this.gDotsElementList.forEach((dotsElement, k) => {
          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
          d3Transition.transition()
            .select(dotsElement)
            .selectAll('.generated-dot')
            .selection()
            .data(gData)
            .transition().duration(SLOW_INTERVAL_MS)
            .attr('cx', (d: number[]) => d[0] * plotSizePx)
            .attr('cy', (d: number[]) => (1.0 - d[1]) * plotSizePx)
            .select('title').text((d: number[], i: number) =>
              `${Number(d[0]).toFixed(2)},${Number(d[1]).toFixed(2)} (${i})`);
        });

        // Move gradients also.
        for (let i = 0; i < gData.length; ++i) {
          gradData[i][0] = gData[i][0];
          gradData[i][1] = gData[i][1];
        }
        gradDotsElementList.forEach((dotsElement, k) => {
          const plotSizePx = k === 0 ? this.plotSizePx : this.smallPlotSizePx;
          const arrowWidth = k === 0 ? 0.002 : 0.001;
          d3Transition.transition()
            .select(dotsElement)
            .selectAll('.gradient-generated').selection().data(gradData)
            .transition().duration(SLOW_INTERVAL_MS)
            .attr('points', (d: number[]) =>
              this.createArrowPolygon(d, plotSizePx, arrowWidth));
        });
      }

      // Simple grid-based evaluation.
      this.evaluator.updateGridsForGenerated(gData);
      this.updateChartData(this.evalChartData, this.iterationCount, [
        this.evaluator.getKLDivergenceScore(),
        this.evaluator.getJSDivergenceScore()
      ]);
      this.evalChart.update();

      if (this.slowMode) {
        await this.sleep(SLOW_INTERVAL_MS);
        this.dehighlightStep();

        const container =
          document.getElementById('model-visualization-container');
        if (container.classList.contains('any-highlighted')) {
          container.classList.remove('any-highlighted');
        }
        document.getElementById(
          'component-discriminator').classList.remove('deactivated');
        document.getElementById(
          'component-g-loss').classList.remove('activated');
        for (let i = 0; i < this.gFlowElements.length; ++i) {
          this.gFlowElements[i].classList.remove('g-activated');
        }
      }
    }

    if (this.iterationCount >= 999999) {
      this.isPlaying = false;
    }
    
    requestAnimationFrame(() => this.iterateTraining(true));
  }

  private createArrowPolygon(d: number[],
    plotSizePx: number, arrowWidth: number) {
    const gradSize = Math.sqrt(
      d[2] * d[2] + d[3] * d[3] + 0.00000001);
    const xNorm = d[2] / gradSize;
    const yNorm = d[3] / gradSize;
    return `${d[0] * plotSizePx},
      ${(1.0 - d[1]) * plotSizePx}
      ${(d[0] - yNorm * (-1) * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * arrowWidth)) * plotSizePx}
      ${(d[0] + d[2] * GRAD_ARROW_UNIT_LEN) * plotSizePx},
      ${(1.0 - (d[1] + d[3] * GRAD_ARROW_UNIT_LEN)) * plotSizePx}
      ${(d[0] - yNorm * arrowWidth) * plotSizePx},
      ${(1.0 - (d[1] - xNorm * (-1) * arrowWidth)) * plotSizePx}`;
  }

  private createGridCellsFromManifoldData(manifoldData: Float32Array[]) {
    const gridData: ManifoldCell[] = [];
    let areaSum = 0.0;
    for (let i = 0; i < NUM_MANIFOLD_CELLS * NUM_MANIFOLD_CELLS; ++i) {
      const x = i % NUM_MANIFOLD_CELLS;
      const y = Math.floor(i / NUM_MANIFOLD_CELLS);
      const index = x + y * (NUM_MANIFOLD_CELLS + 1);

      const gridCell = [];
      gridCell.push(manifoldData[index]);
      gridCell.push(manifoldData[index + 1]);
      gridCell.push(manifoldData[index + 1 + (NUM_MANIFOLD_CELLS + 1)]);
      gridCell.push(manifoldData[index + (NUM_MANIFOLD_CELLS + 1)]);
      gridCell.push(manifoldData[index]);

      // Calculate area by using four points.
      let area = 0.0;
      for (let j = 0; j < 4; ++j) {
        area += gridCell[j % 4][0] * gridCell[(j + 1) % 4][1] -
          gridCell[j % 4][1] * gridCell[(j + 1) % 4][0];
      }
      area = 0.5 * Math.abs(area);
      areaSum += area;

      gridData.push({ points: gridCell, area });
    }
    // Normalize area.
    gridData.forEach(grid => {
      if (grid.area) {
        grid.area = grid.area / areaSum;
      }
    });

    return gridData;
  }

  private playGeneratorAnimation() {
    if (this.noiseSize <= 2) {
      const manifoldData: Float32Array[] = [];
      const numBatches = Math.ceil(Math.pow(
        NUM_MANIFOLD_CELLS + 1, this.noiseSize) / BATCH_SIZE);
      const remainingDummy = BATCH_SIZE * numBatches - Math.pow(
        NUM_MANIFOLD_CELLS + 1, this.noiseSize) * 2;
      for (let k = 0; k < numBatches; ++k) {
        const maniArray: Float32Array =
          this.uniformNoiseProvider.getNextCopy().dataSync() as Float32Array;
        for (let i = 0; i < (k + 1 < numBatches ?
          BATCH_SIZE : BATCH_SIZE - remainingDummy); ++i) {
          if (this.noiseSize >= 2) {
            manifoldData.push(maniArray.slice(i * 2, i * 2 + 2));
          } else {
            manifoldData.push(new Float32Array([maniArray[i], 0.5]));
          }
        }
      }

      // Create grid cells.
      const noiseData = this.noiseSize === 1
        ? [{ points: manifoldData }]
        : this.createGridCellsFromManifoldData(manifoldData);

      const gridData = d3.select('#svg-generator-manifold')
        .selectAll('.grids').data();

      const uniformDotsData = d3.select('#svg-generator-manifold')
        .selectAll('.uniform-generated-dot').data();

      const manifoldCell =
        line()
          .x((d: number[]) => d[0] * this.mediumPlotSizePx)
          .y((d: number[]) => (1.0 - d[1]) * this.mediumPlotSizePx);

      // Visualize noise.
      d3.select('#svg-generator-manifold')
        .selectAll('.grids')
        .data(noiseData)
        .select('.manifold-cell')
        .attr('d', (d: ManifoldCell) => manifoldCell(
          d.points.map(point => [point[0], point[1]] as [number, number])
        ))
        .style('fill-opacity', (d: ManifoldCell, i: number) => {
          return this.selectedNoiseType === '2D Gaussian'
            ? Math.min(0.1 + this.densitiesForGaussian[i] /
              (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.2, 0.9)
            : (this.noiseSize === 2 ? Math.max(
              0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
              : 'none');
        });

      if (this.noiseSize === 1) {
        d3.select('#svg-generator-manifold')
          .selectAll('.uniform-generated-dot')
          .data(manifoldData)
          .attr('cx', (d: Float32Array) => d[0] * this.mediumPlotSizePx)
          .attr('cy', (d: Float32Array) =>
            (1.0 - d[1]) * this.mediumPlotSizePx);
      }

      // Transition to current manifold.
      d3Transition.transition()
        .select('#svg-generator-manifold')
        .selectAll('.grids')
        .selection()
        .data(gridData)
        .transition().duration(2000)
        .select('.manifold-cell')
        .attr('d', (d: ManifoldCell) => manifoldCell(
          d.points.map(point => [point[0], point[1]] as [number, number])
        ))
        .style('fill-opacity', (d: ManifoldCell, i: number) => {
          return this.selectedNoiseType === '2D Gaussian'
            ? Math.min(0.1 + this.densitiesForGaussian[i] /
              (d.area! * Math.pow(NUM_MANIFOLD_CELLS, 2)) * 0.3, 0.9)
            : (this.noiseSize === 2 ? Math.max(
              0.9 - d.area! * 0.4 * Math.pow(NUM_MANIFOLD_CELLS, 2), 0.1)
              : 'none');
        });

      if (this.noiseSize === 1) {
        d3Transition.transition()
          .select('#svg-generator-manifold')
          .selectAll('.uniform-generated-dot')
          .selection()
          .data(uniformDotsData)
          .transition().duration(2000)
          .attr('cx', (d: Float32Array) => d[0] * this.mediumPlotSizePx)
          .attr('cy', (d: Float32Array) =>
            (1.0 - d[1]) * this.mediumPlotSizePx);
      }
    }
  }

  private async highlightStep(isForD: boolean,
    componentElementNames: string[], tooltipElementName: string) {
    await this.sleep(SLOW_INTERVAL_MS);
    this.dehighlightStep();

    this.highlightedComponents =
      componentElementNames.map(componentElementName =>
        document.getElementById(componentElementName) as HTMLDivElement);
    this.highlightedTooltip =
      document.getElementById(tooltipElementName) as HTMLDivElement;

    this.highlightedComponents.forEach(component =>
      component.classList.add('highlighted'));
    this.highlightedTooltip.classList.add('shown');
    this.highlightedTooltip.classList.add('highlighted');

    await this.sleep(SLOW_INTERVAL_MS);
  }

  private dehighlightStep() {
    if (this.highlightedComponents) {
      this.highlightedComponents.forEach(component => {
        component.classList.remove('highlighted');
      });
    }
    if (this.highlightedTooltip) {
      this.highlightedTooltip.classList.remove('shown');
      this.highlightedTooltip.classList.remove('highlighted');
    }
  }

  private async onClickSaveModelButton() {
    const dTensors: tf.NamedTensorMap = 
      this.model.dVariables.reduce((obj, item, i) => {
        obj[`d-${i}`] = item;
        return obj;
      }, {});
    const gTensors: tf.NamedTensorMap = 
      this.model.gVariables.reduce((obj, item, i) => {
        obj[`g-${i}`] = item;
        return obj;
      }, {});
    const tensors: tf.NamedTensorMap = {...dTensors, ...gTensors};

    const modelInfo: {} = {
      'shape_name': this.selectedShapeName,
      'iter_count': this.iterationCount,
      'config': {
        selectedNoiseType: this.selectedNoiseType,
        noiseSize: this.noiseSize,
        numGeneratorLayers: this.numGeneratorLayers,
        numDiscriminatorLayers: this.numDiscriminatorLayers,
        numGeneratorNeurons: this.numGeneratorNeurons,
        numDiscriminatorNeurons: this.numDiscriminatorNeurons,
        dLearningRate: this.dLearningRate,
        gLearningRate: this.gLearningRate,
        dOptimizerType: this.dOptimizerType,
        gOptimizerType: this.gOptimizerType,
        lossType: this.lossType,
        kDSteps: this.kDSteps,
        kGSteps: this.kGSteps,
      }
    };
    const weightDataAndSpecs = await tf.io.encodeWeights(tensors);
    const modelArtifacts: tf.io.ModelArtifacts = {
      modelTopology: modelInfo,
      weightSpecs: weightDataAndSpecs.specs,
      weightData: weightDataAndSpecs.data,
    };

    const downloadTrigger = 
      tf.io.getSaveHandlers('downloads://ganlab_trained_model')[0];
    await downloadTrigger.save(modelArtifacts);
  }

  private async loadPretrainedWeightFile(filename: string): 
      Promise<tf.io.ModelArtifacts> {
    const handler = 
      tf.io.browserHTTPRequest(`pretrained_models/${filename}.json`);
    const loadedModel: tf.io.ModelArtifacts = await handler.load();

    this.iterationCount = loadedModel.modelTopology['iter_count'];
    
    const loadedConfig: {} = loadedModel.modelTopology['config'];
    for (let configProperty in loadedConfig) {
      this[configProperty] = loadedConfig[configProperty];
    }

    document.getElementById('num-g-layers')!.innerText =
      this.numGeneratorLayers.toString();
    document.getElementById('num-d-layers')!.innerText =
      this.numDiscriminatorLayers.toString();
    document.getElementById('num-g-neurons')!.innerText =
      this.numGeneratorNeurons.toString();
    document.getElementById('num-d-neurons')!.innerText =
      this.numDiscriminatorNeurons.toString();
    document.getElementById('k-d-steps')!.innerText = this.kDSteps.toString();
    document.getElementById('k-g-steps')!.innerText = this.kGSteps.toString();

    return loadedModel as Promise<tf.io.ModelArtifacts>;
  }

  private recreateCharts() {
    document.getElementById('chart-container').style.visibility = 'hidden';

    this.costChartData = new Array<ChartData>(2);
    for (let i = 0; i < this.costChartData.length; ++i) {
      this.costChartData[i] = [];
    }
    if (this.costChart != null) {
      this.costChart.destroy();
    }
    const costChartSpecification = [
      { label: 'Discriminator\'s Loss', color: 'rgba(5, 117, 176, 0.5)' },
      { label: 'Generator\'s Loss', color: 'rgba(123, 50, 148, 0.5)' }
    ];
    this.costChart = this.createChart(
      'cost-chart', this.costChartData, costChartSpecification, 0);

    this.evalChartData = new Array<ChartData>(2);
    for (let i = 0; i < this.evalChartData.length; ++i) {
      this.evalChartData[i] = [];
    }
    if (this.evalChart != null) {
      this.evalChart.destroy();
    }
    const evalChartSpecification = [
      { label: 'KL Divergence (by grid)', color: 'rgba(220, 80, 20, 0.5)' },
      { label: 'JS Divergence (by grid)', color: 'rgba(200, 150, 10, 0.5)' }
    ];
    this.evalChart = this.createChart(
      'eval-chart', this.evalChartData, evalChartSpecification, 0);
  }

  private updateChartData(data: ChartData[][], xVal: number, yList: number[]) {
    for (let i = 0; i < yList.length; ++i) {
      data[i].push({ x: xVal, y: yList[i].toFixed(3) });
    }
  }

  private createChart(
    canvasId: string, chartData: ChartData[][],
    specification: Array<{ label: string, color: string }>,
    min?: number, max?: number): Chart {
    const context = (document.getElementById(canvasId) as HTMLCanvasElement)
      .getContext('2d') as CanvasRenderingContext2D;
    const chartDatasets = specification.map((chartSpec, i) => {
      return {
        data: chartData[i],
        backgroundColor: chartSpec.color,
        borderColor: chartSpec.color,
        borderWidth: 1,
        fill: false,
        label: chartSpec.label,
        lineTension: 0,
        pointHitRadius: 8,
        pointRadius: 0
      };
    });

    return new Chart(context, {
      type: 'line',
      data: { datasets: chartDatasets },
      options: {
        animation: { duration: 0 },
        legend: {
          labels: { boxWidth: 10 }
        },
        responsive: false,
        scales: {
          xAxes: [{ type: 'linear', position: 'bottom' }],
          yAxes: [{ ticks: { max, min } }]
        }
      }
    });
  }

  private sleep(ms: number) {
    return new Promise(resolve => {
      const check = () => {
        if (this.isPlaying) {
          this.isPausedOngoingIteration = false;
          resolve();
        } else {
          this.isPausedOngoingIteration = true;
          setTimeout(check, 1000);
        }
      };
      setTimeout(check, ms);
    });
  }
}

document.registerElement(GANLab.prototype.is, GANLab);
