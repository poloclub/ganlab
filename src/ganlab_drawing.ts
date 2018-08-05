export class GANLabDrawing {
  private _drawingPositions: Array<[number, number]>;
  private isDrawing: boolean;
  private context: CanvasRenderingContext2D;

  constructor(private canvas: HTMLCanvasElement, private plotSizePx: number) {
    this._drawingPositions = [];
    this.isDrawing = false;

    this.context = canvas.getContext('2d');
    this.context.strokeStyle = 'rgba(0, 136, 55, 0.25)';
    this.context.lineJoin = 'round';
    this.context.lineWidth = 10;
    const drawingContainer =
      document.getElementById('vis-content-container') as HTMLDivElement;
    const offsetLeft = drawingContainer.offsetLeft + 5;
    const offsetTop = drawingContainer.offsetTop + 15;

    this.canvas.addEventListener('mousedown', (event: MouseEvent) => {
      this.isDrawing = true;
      this.draw([event.pageX - offsetLeft, event.pageY - offsetTop]);
    });
    this.canvas.addEventListener('mousemove', (event: MouseEvent) => {
      if (this.isDrawing) {
        this.draw([event.pageX - offsetLeft, event.pageY - offsetTop]);
      }
    });
    this.canvas.addEventListener('mouseup', (event: Event) => {
      this.isDrawing = false;
    });
  }

  get drawingPositions(): Array<[number, number]> {
    return this._drawingPositions;
  }

  prepareDrawing() {
    this._drawingPositions = [];
    this.context.clearRect(
      0, 0, this.context.canvas.width, this.context.canvas.height);
    const drawingElement =
      document.getElementById('drawing-container') as HTMLElement;
    drawingElement.style.display = 'block';
    const drawingBackgroundElement =
      document.getElementById('drawing-disable-background') as HTMLDivElement;
    drawingBackgroundElement.style.display = 'block';
  }

  private draw(position: [number, number]) {
    this._drawingPositions.push(
      [position[0] / this.plotSizePx, 1.0 - position[1] / this.plotSizePx]);
    this.context.beginPath();
    this.context.moveTo(position[0] - 1, position[1]);
    this.context.lineTo(position[0], position[1]);
    this.context.closePath();
    this.context.stroke();
  }
}
