import { Physics } from "./Physics";

export class Renderer
{
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private canvasFormat!: GPUTextureFormat;

    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private uniformBuffer!: GPUBuffer;

    private uniforms!: Float32Array;

    private hairStateStorage!: GPUBuffer[]

    private shaderModule!: GPUShaderModule;
    private bindGroupLayout!: GPUBindGroupLayout;
    private bindGroup!: GPUBindGroup[];
    private pipeline!: GPURenderPipeline;
    private renderPassDescriptor!: GPURenderPassDescriptor;

    private compute_shaderModule!: GPUShaderModule;
    private compute_bindGroupLayout!: GPUBindGroupLayout;
    private compute_bindGroup!: GPUBindGroup[];
    private compute_pipeline!: GPUComputePipeline;

    private step: boolean = false;

    private readonly numHairStrands = 100.0;
    
    // TODO: Is the vertex buffer redundant?
    // Vertex and index data
    private readonly strandVertices = new Float32Array(
    [
        //   X, Y, Z
        0.0, 0.0, 2.8,
        -0.2, -0.2, 2.8,
        -0.4, -0.4, 2.8,
        -0.6, -0.6, 2.8,
        // -0.0, -0.2, 2.8,
        // -0.0, -0.25, 2.8,
        // -0.0, -0.3, 2.8,
        // -0.0, -0.35, 2.8,
        // -0.0, -0.4, 2.8,
        // -0.0, -0.45, 2.8,
        // -0.0, -0.5, 2.8,
        // -0.0, -0.55, 2.8,
        // -0.0, -0.6, 2.8,
        // -0.0, -0.65, 2.8,
        // -0.0, -0.7, 2.8,
        // -0.0, -0.75, 2.8,
    ]);

    private readonly indices = new Uint16Array(
        Array.from({ length: this.strandVertices.length / 3 }, (_, i) => i));

    // Keeping everything in one class for now
    // TODO: Add this later
    // private m_physics!: Physics;

    constructor(private canvas: HTMLCanvasElement) {}

    public async init()
    {
        // Setup
        await this.getGPU();
        this.connectCanvas();

        // Pipeline
        this.loadShaders();
        this.loadComputeShader();
        this.createBuffers();
        this.createPipeline();
        this.compute_createPipeline();

        // Render
        this.createRenderPassDescriptor();
        this.render();
    }

    // Setup
    private async getGPU()
    {
        // Check if webgpu is supported
        if (!navigator.gpu)
        {
            this.onError("WebGPU not supported on this browser.");
            return;
        }

        // Get GPU adapter with a preference for high performance/discrete GPUs
        const adapter: GPUAdapter | null = await navigator.gpu.requestAdapter(
        {
            powerPreference: "high-performance"
        });

        if (!adapter)
        {
            this.onError("No GPU Adapter found.");
            return;
        }

        // Get logical interface
        this.device = await adapter.requestDevice();
        // Capture all errors
        this.device.addEventListener('uncapturederror', event => console.log(event.error.message));
    }

    private connectCanvas()
    {
        // Connect canvas with GPU interface
        const context = this.canvas.getContext("webgpu");

        if (!context)
        {
            this.onError("Failed to get canvas context :/");
            return;
        }

        this.context = context;

        this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure(
        {
            device: this.device,
            format: this.canvasFormat // texture format
        });
    }

    // Pipeline
    private loadShaders()
    {
        // Vertex and Fragment shaders
        this.shaderModule = this.device.createShaderModule(
        {
            label: "Hair shader",
            code:
            /* wgsl */ `
            @group(0) @binding(0) var<uniform> sceneData: SceneData;
            @group(0) @binding(1) var<storage> positions: array<f32>;

            struct Ray
            {
                origin: vec3f,
                dir: vec3f
            }

            struct Camera
            {
                location: vec3f,
                focalLength: f32,
                imageDimensions: vec2f,
                filmPlaneDimensions: vec2f
            };

            struct SceneData
            {
                resolution: vec2f,
                numStrandVertices: f32
            };

            fn viewTransformMatrix(eye: vec3f, lookAt: vec3f, up: vec3f) 
                                    -> mat4x4<f32>
            {
                var forward = normalize(lookAt - eye);
                var right = normalize(cross(up, forward));
                var u = normalize(cross(forward, right));

                return mat4x4<f32>(
                    right.x, u.x, forward.x, 0.0f,
                    right.y, u.y, forward.y, 0.0f,
                    right.z, u.z, forward.z, 0.0f,
                    -dot(eye, right), -dot(eye, u), -dot(eye, forward), 1.0f
                );
            }

            fn projectionMatrix(angle: f32, aspect_ratio: f32, near: f32, far: f32) -> mat4x4<f32>
            {
                let a: f32 = 1.0 / tan(radians(angle/2));
                let m1 = far/(far - near);
                let m2 = -near * far/(far -near);

                return mat4x4<f32>(
                    vec4<f32>(a * aspect_ratio, 0.0, 0.0, 0.0),
                    vec4<f32>(0.0, a, 0.0, 0.0),
                    vec4<f32>(0.0f, 0.0f, m1, 1.0),
                    vec4<f32>(0.0, 0.0, m2, 0.0)
                );
            }

            struct VertReturn
            {
                @builtin(position) pos : vec4f,
                @location(0) dir: vec3f
            }

            @vertex
            fn vertexMain(@builtin(instance_index) instance: u32,
                            @builtin(vertex_index) vert_idx: u32)
                -> VertReturn
            {
                // Get pos from storage buffer                    
                let i: f32 = f32(instance);
                let numStrandVertices: f32 = sceneData.numStrandVertices;

                // vertex index is indicative of position of particle within a hair strand
                // TODO: Is there a better way to get this data? Uniform buffer or something?
                var pos: vec4f = vec4f(
                            positions[u32(i * numStrandVertices) + vert_idx * 3],
                            positions[u32(i * numStrandVertices) + vert_idx * 3 + 1],
                            positions[u32(i * numStrandVertices) + vert_idx * 3 + 2],
                            1.0f);

                var cam: Camera;
                cam.imageDimensions = sceneData.resolution;

                // meters
                cam.filmPlaneDimensions = vec2f(25.0f, 25.0f);

                let fov = 30.0f;
                cam.focalLength = (cam.filmPlaneDimensions.y / 2.0f)/tan(radians(fov/2));

                cam.location = vec3f(0.8f, 0.6f, 0.0f);

                let lookAt: vec3f = vec3f(0.0f, 0.0f, 2.8f);

                var view: mat4x4<f32> = viewTransformMatrix(
                    cam.location,
                    lookAt,
                    vec3f(0.0f, 1.0f, 0.0f)
                );

                let angle = 30.0f;

                // arbitrary far clip plane for now
                var projection : mat4x4<f32> = projectionMatrix(angle, sceneData.resolution.y/sceneData.resolution.x, 1.0f, 100.0f);
                var result: vec4f = projection * view * pos;
            
                // TODO: Clean code and var names
                var pos2: vec3f;
               
                if ( vert_idx < u32(numStrandVertices - 1.0) )
                {
                    pos2 = vec3f(
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3],
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3 + 1],
                        positions[u32(i * numStrandVertices) + (vert_idx + 1) * 3 + 2]);
                }
                else
                {
                    pos2 = pos.xyz;
                    pos = vec4f(
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3],
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3 + 1],
                        positions[u32(i * numStrandVertices) + (vert_idx - 1) * 3 + 2],
                        1.0f);
                }

                
                var returnVal: VertReturn;
                returnVal.pos = result;
                returnVal.dir = normalize(pos.xyz - pos2);

                return returnVal;
            }

            @fragment
            fn fragmentMain(input: VertReturn)
                -> @location(0) vec4f
            {
                let lightDir = normalize(vec3(1.0, 1.0, 0.0));
                let cosT: f32 = sqrt(1.0 - pow(dot(lightDir, input.dir), 2));

                let lightColor = vec3f(1.0, 1.0, 1.0);

                return vec4f(lightColor * cosT, 1.0f);
            }
            `
        });
    }

    private loadComputeShader()
    {
        this.compute_shaderModule = this.device.createShaderModule(
        {
            label: "Hair simulation shader",
            code: 
            /* wgsl */ `
                @group(0) @binding(0) var<uniform> sceneData: SceneData;
                @group(0) @binding(1) var<storage> positionsIn: array<f32>;
                @group(0) @binding(2) var<storage> velocitiesIn: array<f32>;
                @group(0) @binding(3) var<storage, read_write> positionsOut: array<f32>;
                @group(0) @binding(4) var<storage, read_write> velocitiesOut: array<f32>;

                struct SceneData
                {
                    resolution: vec2f,
                    numStrandVertices: f32
                };


                const mass = 0.1f;
                const gravity : f32 = -9.8f;
                const deltaTime : f32 = 1.0f/60.0f;

                const damping = 0.0f;
                const k = 40.0f;
                const rest_length = 0.2f; // TODO: Don't hardcode

                // TODO: Why is the force reducing over time even when particles are in the same position?
                fn calculateForces(idx: u32, last_vertex: bool) -> vec3<f32>
                {
                    let vi : vec3<f32> = vec3(velocitiesIn[idx], velocitiesIn[idx + 1],
                                            velocitiesIn[idx + 2]);

                    let curr_pos : vec3<f32> = vec3(positionsIn[idx], positionsIn[idx + 1],
                                                 positionsIn[idx + 2]);
                    let prev_pos : vec3<f32> = vec3(positionsIn[idx  - 3], positionsIn[idx - 2],
                                                positionsIn[idx - 1]);

                    let length1 : f32 = length(curr_pos - prev_pos);
                    let dir1 : vec3<f32> = normalize(prev_pos - curr_pos);
                    
                    // Spring force towards previous strand
                    var force : vec3<f32> = dir1 * (length1 - rest_length) * k;

                    force += vi * damping;

                    force.y += mass * gravity;

                    if (!last_vertex)
                    {
                        let next_pos : vec3<f32> = vec3(positionsIn[idx + 3], positionsIn[idx + 4],
                                                        positionsIn[idx + 5]
                                                    );

                        let length2: f32 = length(curr_pos - next_pos);
                        let dir2 : vec3<f32> = normalize(curr_pos - next_pos);
                        
                        // Spring force towards next strand
                        force += dir2 * (length2 - rest_length) * k;

                        let v_last : vec3<f32> = vec3(velocitiesIn[idx + 3], velocitiesIn[idx + 4],
                                                    velocitiesIn[idx + 5]
                                                );
                        
                        force += -v_last * damping;
                    }
                    
                    return force;
                }

                @compute
                @workgroup_size(8) // TODO: Don't hard code workgroup size
                fn computeMain(@builtin(global_invocation_id) id: vec3<u32>)
                {                   
                    let idx = id.x;

                    let numStrandVertices = sceneData.numStrandVertices;

                    let vert_idx = f32(idx % u32(numStrandVertices));

                    if ( vert_idx > 2.0f && idx % 3 == 0 )
                    {
                        let force: vec3<f32> = calculateForces(idx, vert_idx >= numStrandVertices - 3.0f);
                        let acceleration: vec3<f32> = force / mass;
                        // let acceleration = vec3f(0.0f, 0.0f, 0.0f);

                        // TODO: Do we even need to store velocities?
                        // Maybe for some force/damping?
                        velocitiesOut[idx] = acceleration.x * deltaTime;
                        velocitiesOut[idx + 1] = acceleration.y * deltaTime;
                        velocitiesOut[idx + 2] = acceleration.z * deltaTime;

                        positionsOut[idx] = positionsIn[idx] + velocitiesOut[idx] * deltaTime;
                        positionsOut[idx + 1] = positionsIn[idx + 1] + velocitiesOut[idx + 1] * deltaTime;
                        positionsOut[idx + 2] = positionsIn[idx + 2] + velocitiesOut[idx + 2] * deltaTime;
                    }
                }
            `
        });
    }

    private createBuffers()
    {        
        // Vertex Buffer
        this.vertexBuffer = this.device.createBuffer(
        {
            label: "Strand vertices",
            size: this.strandVertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });

        // Index Buffer
        this.indexBuffer = this.device.createBuffer(
        {
            label: "Strand indices",
            size: this.indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
        });

        // Uniform buffer
        this.uniforms = new Float32Array(3);

        // Resolution
        this.uniforms[0] = this.canvas.width;
        this.uniforms[1] = this.canvas.height;
        this.uniforms[2] = this.strandVertices.length;
        
        this.uniformBuffer = this.device.createBuffer(
        {
            label: "Uniform buffer",
            size: Math.ceil(this.uniforms.byteLength / 16) * 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        // Write buffers
        this.device.queue.writeBuffer(this.vertexBuffer, 0, this.strandVertices);
        this.device.queue.writeBuffer(this.indexBuffer, 0, this.indices);
        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniforms);

        const radius = 0.5;
        const scalpCenterX = 0.0;
        const scalpCenterY = 0.5;

        // Storage Buffers
        const positionsArray = new Float32Array(this.numHairStrands * this.strandVertices.length);
        const velocitiesArray = new Float32Array(this.numHairStrands * this.strandVertices.length);

        this.hairStateStorage = [
            this.device.createBuffer(
            {
                label: "Positions",
                size: positionsArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            this.device.createBuffer(
            {
                label: "Velocities",
                size: velocitiesArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            this.device.createBuffer(
            {
                label: "PositionsCopy",
                size: positionsArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
            this.device.createBuffer(
            {
                label: "VelocitiesCopy",
                size: velocitiesArray.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            }),
        ];

        this.device.queue.writeBuffer(this.hairStateStorage[1], 0, velocitiesArray);
        this.device.queue.writeBuffer(this.hairStateStorage[3], 0, velocitiesArray);
        
        // fill positions
        for (let ii = 0; ii < this.numHairStrands; ii++)
        {
            for (let jj = 0; jj < this.strandVertices.length; jj += 3)
            {
                let theta = ii * Math.PI/(this.numHairStrands - 1.0);
                let base_idx = ii * this.strandVertices.length + jj;

                positionsArray[base_idx] = radius * Math.cos(theta) + scalpCenterX + this.strandVertices[jj];
                positionsArray[base_idx + 1] = -1.0 * radius * Math.sin(theta) + scalpCenterY + this.strandVertices[jj + 1];
                positionsArray[base_idx + 2] = this.strandVertices[jj + 2];
            }
        }

        this.device.queue.writeBuffer(this.hairStateStorage[0], 0, positionsArray);
        this.device.queue.writeBuffer(this.hairStateStorage[2], 0, positionsArray);
    }

    private createPipeline()
    {
        // Vertex Buffer Layout
        const vertexBufferLayout : GPUVertexBufferLayout =
        {
            // 3 values per vertex (x, y, z)
            arrayStride: 12,
            attributes:
            [{
                format: "float32x3",
                offset: 0,
                shaderLocation: 0
            }]
        };

        this.bindGroupLayout = this.device.createBindGroupLayout(
        {
            label: "Hair Group Layout Vertex",
            entries: 
            [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: {} // Hair uniform buffer
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: "read-only-storage"} // Hair positions buffer
            }]
        });

        // TODO: Should this be done here or elsewhere?
        this.bindGroup = [
            this.device.createBindGroup(
            {
                label: "Vertex Bind group A",
                layout: this.bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformBuffer }
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.hairStateStorage[0] }
                    }
                ]
            }),
            this.device.createBindGroup(
            {
                label: "Vertex Bind group B",
                layout: this.bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformBuffer }
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.hairStateStorage[2] }
                    }
                ]
            })
        ];

        // Pipeline Layout
        const pipelineLayout = this.device.createPipelineLayout(
        {
            label: "Hair Pipeline Layout Vertex",
            bindGroupLayouts: [ this.bindGroupLayout ]
        });

        // Pipeline
        this.pipeline = this.device.createRenderPipeline(
        {
            label: "Hair pipeline",
            layout: pipelineLayout,
            vertex: {
                module: this.shaderModule,
                entryPoint: "vertexMain",
                buffers: [vertexBufferLayout]
            },
            fragment: {
                module: this.shaderModule,
                entryPoint: "fragmentMain",
                targets: 
                [{
                    format: this.canvasFormat
                }]
            },
            primitive: {
                topology: 'line-strip',
                stripIndexFormat: 'uint16'
            }
        });
    }

    private compute_createPipeline()
    {
        // Create the bind group layout and pipeline layout.
        this.compute_bindGroupLayout = this.device.createBindGroupLayout(
        {
            label: "Hair Bind Group Layout",
            entries: 
            [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {} // Hair uniform buffer
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage"} // Hair positions buffer
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "read-only-storage"} // Hair velocities buffer
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"} // Hair positions buffer
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE,
                buffer: { type: "storage"} // Hair velocities buffer
            }]
        });


        this.compute_bindGroup = [
            this.device.createBindGroup(
            {
                label: "Simulation bind group A",
                layout: this.compute_bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformBuffer }
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.hairStateStorage[0] }
                    },
                    {
                        binding: 2,
                        resource: { buffer: this.hairStateStorage[1] }
                    },
                    {
                        binding: 3,
                        resource: { buffer: this.hairStateStorage[2] }
                    },
                    {
                        binding: 4,
                        resource: { buffer: this.hairStateStorage[3] }
                    }
                ]
            }),
            this.device.createBindGroup(
            {
                label: "Simulation bind group B",
                layout: this.compute_bindGroupLayout,
                entries: [
                    {
                        binding: 0,
                        resource: { buffer: this.uniformBuffer }
                    },
                    {
                        binding: 1,
                        resource: { buffer: this.hairStateStorage[2] }
                    },
                    {
                        binding: 2,
                        resource: { buffer: this.hairStateStorage[3] }
                    },
                    {
                        binding: 3,
                        resource: { buffer: this.hairStateStorage[0] }
                    },
                    {
                        binding: 4,
                        resource: { buffer: this.hairStateStorage[1] }
                    }
                ]
            })
        ];

        const compute_pipelineLayout = this.device.createPipelineLayout(
        {
            label: "Hair Pipeline Layout Compute",
            bindGroupLayouts: [ this.compute_bindGroupLayout ]
        });

        this.compute_pipeline = this.device.createComputePipeline(
        {
            label: "Simulation pipeline",
            layout: compute_pipelineLayout,
            compute:
            {
                module: this.compute_shaderModule,
                entryPoint: "computeMain",
            }
        });
    }

    // Rendering
    private createRenderPassDescriptor()
    {
        this.renderPassDescriptor =
        {
            label: "Render Pass Description",
            colorAttachments:
            [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: [0.2, 0.2, 0.2, 1],
                loadOp: "clear",
                storeOp: "store",
            }]
        };
    }

    private render()
    {
        const bindGroupIdx = Number(this.step);
        // update view
        (this.renderPassDescriptor.colorAttachments as any)[0].view =
            this.context.getCurrentTexture().createView();

        // create command buffer
        const encoder = this.device.createCommandEncoder();

        // compute pass
        const computePass = encoder.beginComputePass();

        computePass.setPipeline(this.compute_pipeline);
        computePass.setBindGroup(0, this.compute_bindGroup[bindGroupIdx]);

        // TODO: workgroup size hardcoded for now
        const workgroupCount = Math.ceil((this.numHairStrands * this.strandVertices.length) / 8);
        computePass.dispatchWorkgroups(workgroupCount);

        computePass.end();

        // renderpass
        const pass = encoder.beginRenderPass(this.renderPassDescriptor);

        pass.setIndexBuffer(this.indexBuffer, "uint16");
        pass.setPipeline(this.pipeline);
        pass.setVertexBuffer(0, this.vertexBuffer);
        pass.setBindGroup(0, this.bindGroup[bindGroupIdx]);
        pass.drawIndexed(this.indices.length, this.numHairStrands);

        pass.end();

        // Finish command buffer and immediately submit it
        this.device.queue.submit([encoder.finish()]);

        this.step = !this.step;

        // Loop every frame
        requestAnimationFrame(() => this.render());
    }


    // Misc
    private onError(msg: string)
    {
        document.body.innerHTML = `<p>${msg}</p>`;
        console.error(msg);
    }
}
