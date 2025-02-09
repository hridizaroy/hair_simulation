import { Renderer } from "./Renderer";

// Get access to canvas
const canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

// Start rendering
const renderer = new Renderer(canvas);
renderer.init();
