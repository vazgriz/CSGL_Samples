using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Diagnostics;

using CSGL;
using CSGL.GLFW;
using CSGL.Vulkan;
using CSGL.GLFW.Unmanaged;

using Image = CSGL.Vulkan.Image;
using Buffer = CSGL.Vulkan.Buffer;

namespace Samples {
    public struct Vertex {
        public Vector3 position;
        public Vector3 color;

        public Vertex(Vector3 position, Vector3 color) {
            this.position = position;
            this.color = color;
        }

        public static VkVertexInputBindingDescription GetBindingDescription() {
            var result = new VkVertexInputBindingDescription();
            result.binding = 0;
            result.stride = (uint)Interop.SizeOf<Vertex>();
            result.inputRate = VkVertexInputRate.Vertex;

            return result;
        }

        public static List<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
            Vertex v = new Vertex();
            var a = new VkVertexInputAttributeDescription();
            a.binding = 0;
            a.location = 0;
            a.format = VkFormat.R32G32B32_Sfloat;
            a.offset = (uint)Interop.Offset(ref v, ref v.position);

            var b = new VkVertexInputAttributeDescription();
            b.binding = 0;
            b.location = 1;
            b.format = VkFormat.R32G32B32_Sfloat;
            b.offset = (uint)Interop.Offset(ref v, ref v.color);

            return new List<VkVertexInputAttributeDescription> { a, b };
        }
    }

    public struct UniformBufferObject {
        public Matrix4x4 model;
        public Matrix4x4 view;
        public Matrix4x4 proj;

        public UniformBufferObject(Matrix4x4 model, Matrix4x4 view, Matrix4x4 proj) {
            this.model = model;
            this.view = view;
            this.proj = proj;
        }
    }

    class Program : IDisposable {
        static void Main(string[] args) {
            using (var p = new Program()) {
                p.Run();
            }
            Console.ReadLine();
        }

        List<string> layers = new List<string> {
            "VK_LAYER_LUNARG_standard_validation",
            //"VK_LAYER_LUNARG_api_dump"
        };

        List<string> deviceExtensions = new List<string> {
            "VK_KHR_swapchain"
        };

        List<string> instanceExtensions = new List<string> {
            "VK_EXT_debug_report"
        };

        Vertex[] vertices = {
            new Vertex(new Vector3(-0.5f, -0.5f, 0), new Vector3(1, 0, 0)),
            new Vertex(new Vector3(0.5f, -0.5f, 0), new Vector3(0, 1, 0)),
            new Vertex(new Vector3(0.5f, 0.5f, 0), new Vector3(0, 0, 1)),
            new Vertex(new Vector3(-0.5f, 0.5f, 0), new Vector3(1, 1, 1)),
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0
        };

        int width = 800;
        int height = 600;
        WindowPtr window;

        uint graphicsIndex;
        uint presentIndex;
        Queue graphicsQueue;
        Queue presentQueue;

        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;

        Instance instance;
        DebugReportCallback debugCallbacks;
        Surface surface;
        PhysicalDevice physicalDevice;
        Device device;
        Swapchain swapchain;
        List<Image> swapchainImages;
        List<ImageView> swapchainImageViews;
        RenderPass renderPass;
        DescriptorSetLayout descriptorSetLayout;
        PipelineLayout pipelineLayout;
        Pipeline pipeline;
        List<Framebuffer> swapchainFramebuffers;
        CommandPool commandPool;
        Buffer vertexBuffer;
        DeviceMemory vertexBufferMemory;
        Buffer indexBuffer;
        DeviceMemory indexBufferMemory;
        Buffer uniformBuffer;
        DeviceMemory uniformBufferMemory;
        DescriptorPool descriptorPool;
        DescriptorSet descriptorSet;
        List<CommandBuffer> commandBuffers;
        Semaphore imageAvailableSemaphore;
        Semaphore renderFinishedSemaphore;

        Stopwatch watch;

        bool recreateSwapchainFlag;

        void Run() {
            GLFW.Init();
            CreateWindow();
            CreateInstance();
            CreateDebugCallbacks();
            CreateSurface();
            PickPhysicalDevice();
            PickQueues();
            CreateDevice();
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateDescriptorSetLayout();
            CreateGraphicsPipeline();
            CreateFramebuffers();
            CreateCommandPool();
            CreateVertexBuffer();
            CreateIndexBuffer();
            CreateUniformBuffer();
            CreateDescriptorPool();
            CreateDescriptorSet();
            CreateCommandBuffers();
            CreateSemaphores();

            MainLoop();
        }

        public void Dispose() {
            imageAvailableSemaphore.Dispose();
            renderFinishedSemaphore.Dispose();
            descriptorPool.Dispose();
            uniformBufferMemory.Dispose();
            uniformBuffer.Dispose();
            indexBufferMemory.Dispose();
            indexBuffer.Dispose();
            vertexBufferMemory.Dispose();
            vertexBuffer.Dispose();
            commandPool.Dispose();
            foreach (var fb in swapchainFramebuffers) fb.Dispose();
            pipeline.Dispose();
            pipelineLayout.Dispose();
            descriptorSetLayout.Dispose();
            renderPass.Dispose();
            foreach (var iv in swapchainImageViews) iv.Dispose();
            swapchain.Dispose();
            device.Dispose();
            surface.Dispose();
            debugCallbacks.Dispose();
            instance.Dispose();
            GLFW.DestroyWindow(window);
            GLFW.Terminate();
        }

        void MainLoop() {
            var waitSemaphores = new List<Semaphore> { imageAvailableSemaphore };
            var waitStages = new List<VkPipelineStageFlags> { VkPipelineStageFlags.ColorAttachmentOutputBit };
            var signalSemaphores = new List<Semaphore> { renderFinishedSemaphore };
            var swapchains = new List<Swapchain> { swapchain };

            var commandBuffer = new List<CommandBuffer> { null };
            var index = new List<uint> { 0 };

            var submitInfo = new SubmitInfo();
            submitInfo.waitSemaphores = waitSemaphores;
            submitInfo.waitDstStageMask = waitStages;
            submitInfo.commandBuffers = commandBuffer;
            submitInfo.signalSemaphores = signalSemaphores;

            var presentInfo = new PresentInfo();
            presentInfo.waitSemaphores = signalSemaphores;
            presentInfo.swapchains = swapchains;
            presentInfo.imageIndices = index;

            var submitInfos = new List<SubmitInfo> { submitInfo };

            GLFW.ShowWindow(window);

            watch = new Stopwatch();
            watch.Start();

            while (true) {
                GLFW.PollEvents();
                if (GLFW.GetKey(window, CSGL.Input.KeyCode.Enter) == CSGL.Input.KeyAction.Press) {
                    break;
                }
                if (GLFW.WindowShouldClose(window)) break;

                UpdateUniformBuffer();

                if (recreateSwapchainFlag) {
                    recreateSwapchainFlag = false;
                    RecreateSwapchain();
                }

                uint imageIndex;
                var result = swapchain.AcquireNextImage(ulong.MaxValue, imageAvailableSemaphore, null, out imageIndex);

                if (result == VkResult.ErrorOutOfDateKhr || result == VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                    continue;
                }

                commandBuffer[0] = commandBuffers[(int)imageIndex];
                swapchains[0] = swapchain;
                index[0] = imageIndex;

                graphicsQueue.Submit(submitInfos, null);
                result = presentQueue.Present(presentInfo);

                if (result == VkResult.ErrorOutOfDateKhr || result == VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                }
            }

            device.WaitIdle();
        }

        void UpdateUniformBuffer() {
            double time = watch.Elapsed.TotalSeconds;

            var ubo = new UniformBufferObject();
            ubo.model = Matrix4x4.CreateRotationZ((float)(time * Math.PI / 2));
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(2, 2, 2),
                new Vector3(0, 0, 0),
                new Vector3(0, 0, 1));
            ubo.proj = Matrix4x4.CreatePerspectiveFieldOfView((float)Math.PI / 4,
                swapchainExtent.width / (float)swapchainExtent.height, 0.1f, 10f);
            ubo.proj.M22 *= -1;

            ulong size = (ulong)Interop.SizeOf<UniformBufferObject>();

            var data = uniformBufferMemory.Map(0, size);
            Interop.Copy(new UniformBufferObject[] { ubo }, data);
            uniformBufferMemory.Unmap();
        }

        void RecreateSwapchain() {
            device.WaitIdle();
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicsPipeline();
            CreateFramebuffers();
            CreateCommandBuffers();
        }

        void OnWindowResized(WindowPtr window, int width, int height) {
            if (width == 0 || height == 0) return;
            recreateSwapchainFlag = true;
        }

        void CreateWindow() {
            GLFW.WindowHint(WindowHint.ClientAPI, (int)ClientAPI.NoAPI);
            GLFW.WindowHint(WindowHint.Visible, 0);
            window = GLFW.CreateWindow(width, height, "Uniform Buffer", MonitorPtr.Null, WindowPtr.Null);
        }

        void CreateInstance() {
            var extensions = new List<string>(GLFW.GetRequiredInstanceExceptions());
            foreach (var extension in instanceExtensions) {
                extensions.Add(extension);
            }

            var appInfo = new ApplicationInfo {
                apiVersion = new VkVersion(1, 0, 0),
                applicationVersion = new VkVersion(1, 0, 0),
                engineVersion = new VkVersion(1, 0, 0),
                applicationName = "Uniform Buffer",
            };

            var info = new InstanceCreateInfo {
                applicationInfo = appInfo,
                extensions = extensions,
                layers = layers
            };
            instance = new Instance(info);
        }

        void DebugCallback(
            VkDebugReportFlagsEXT flags,
            VkDebugReportObjectTypeEXT objectType,
            ulong _object, ulong location,
            int messageCode, string layerPrefix, string message) {

            string type = flags.ToString();
            type = type.Substring(0, type.Length - 6);  //strip "BitExt"

            Console.WriteLine("[{0}] {1}", type, message);
        }

        void CreateDebugCallbacks() {
            DebugReportCallbackCreateInfo info = new DebugReportCallbackCreateInfo {
                callback = DebugCallback,
                flags = VkDebugReportFlagsEXT.DebugBitExt |
                        VkDebugReportFlagsEXT.ErrorBitExt |
                        VkDebugReportFlagsEXT.InformationBitExt |
                        VkDebugReportFlagsEXT.PerformanceWarningBitExt |
                        VkDebugReportFlagsEXT.WarningBitExt
            };

            debugCallbacks = new DebugReportCallback(instance, info);
        }

        void PickPhysicalDevice() {
            physicalDevice = instance.PhysicalDevices[0];
        }

        void CreateSurface() {
            surface = new Surface(instance, window);
        }

        void PickQueues() {
            int g = -1;
            int p = -1;

            for (int i = 0; i < physicalDevice.QueueFamilies.Count; i++) {
                var family = physicalDevice.QueueFamilies[i];
                if ((family.Flags & VkQueueFlags.GraphicsBit) != 0) {
                    g = i;
                }

                if (family.SurfaceSupported(surface)) {
                    p = i;
                }
            }

            graphicsIndex = (uint)g;
            presentIndex = (uint)p;
        }

        void CreateDevice() {
            var features = physicalDevice.Features;

            HashSet<uint> uniqueIndices = new HashSet<uint> { graphicsIndex, presentIndex };
            List<float> priorities = new List<float> { 1f };
            List<DeviceQueueCreateInfo> queueInfos = new List<DeviceQueueCreateInfo>(uniqueIndices.Count);

            int i = 0;
            foreach (var ind in uniqueIndices) {
                var queueInfo = new DeviceQueueCreateInfo {
                    queueFamilyIndex = ind,
                    queueCount = 1,
                    priorities = priorities
                };

                queueInfos.Add(queueInfo);
                i++;
            }

            var info = new DeviceCreateInfo {
                extensions = deviceExtensions,
                queueCreateInfos = queueInfos,
                features = features
            };
            device = new Device(physicalDevice, info);

            graphicsQueue = device.GetQueue(graphicsIndex, 0);
            presentQueue = device.GetQueue(presentIndex, 0);
        }

        SwapchainSupport GetSwapchainSupport(PhysicalDevice physicalDevice) {
            var cap = surface.GetCapabilities(physicalDevice);
            var formats = surface.GetFormats(physicalDevice);
            var modes = surface.GetPresentModes(physicalDevice);

            return new SwapchainSupport(cap, new List<VkSurfaceFormatKHR>(formats), new List<VkPresentModeKHR>(modes));
        }

        VkSurfaceFormatKHR ChooseSwapSurfaceFormat(List<VkSurfaceFormatKHR> formats) {
            if (formats.Count == 1 && formats[0].format == VkFormat.Undefined) {
                var result = new VkSurfaceFormatKHR();
                result.format = VkFormat.B8G8R8A8_Unorm;
                result.colorSpace = VkColorSpaceKHR.SrgbNonlinearKhr;
                return result;
            }

            foreach (var f in formats) {
                if (f.format == VkFormat.B8G8R8A8_Unorm && f.colorSpace == VkColorSpaceKHR.SrgbNonlinearKhr) {
                    return f;
                }
            }

            return formats[0];
        }

        VkPresentModeKHR ChooseSwapPresentMode(List<VkPresentModeKHR> modes) {
            foreach (var m in modes) {
                if (m == VkPresentModeKHR.MailboxKhr) {
                    return m;
                }
            }

            return VkPresentModeKHR.FifoKhr;
        }

        VkExtent2D ChooseSwapExtent(ref VkSurfaceCapabilitiesKHR cap) {
            if (cap.currentExtent.width != uint.MaxValue) {
                return cap.currentExtent;
            } else {
                var extent = new VkExtent2D();
                extent.width = (uint)width;
                extent.height = (uint)height;

                extent.width = Math.Max(cap.minImageExtent.width, Math.Min(cap.maxImageExtent.width, extent.width));
                extent.height = Math.Max(cap.minImageExtent.height, Math.Min(cap.maxImageExtent.height, extent.height));

                return extent;
            }
        }

        void CreateSwapchain() {
            var support = GetSwapchainSupport(physicalDevice);
            var cap = support.cap;

            var surfaceFormat = ChooseSwapSurfaceFormat(support.formats);
            var mode = ChooseSwapPresentMode(support.modes);
            var extent = ChooseSwapExtent(ref cap);

            uint imageCount = cap.minImageCount + 1;
            if (cap.maxImageCount > 0 && imageCount > cap.maxImageCount) {
                imageCount = cap.maxImageCount;
            }

            var oldSwapchain = swapchain;
            var info = new SwapchainCreateInfo();
            info.surface = surface;
            info.oldSwapchain = oldSwapchain;
            info.minImageCount = imageCount;
            info.imageFormat = surfaceFormat.format;
            info.imageColorSpace = surfaceFormat.colorSpace;
            info.imageExtent = extent;
            info.imageArrayLayers = 1;
            info.imageUsage = VkImageUsageFlags.ColorAttachmentBit;

            var queueFamilyIndices = new List<uint> { graphicsIndex, presentIndex };

            if (graphicsIndex != presentIndex) {
                info.imageSharingMode = VkSharingMode.Concurrent;
                info.queueFamilyIndices = queueFamilyIndices;
            } else {
                info.imageSharingMode = VkSharingMode.Exclusive;
            }

            info.preTransform = cap.currentTransform;
            info.compositeAlpha = VkCompositeAlphaFlagsKHR.OpaqueBitKhr;
            info.presentMode = mode;
            info.clipped = true;

            swapchain = new Swapchain(device, info);
            oldSwapchain?.Dispose();

            swapchainImages = new List<Image>(swapchain.Images);

            swapchainImageFormat = surfaceFormat.format;
            swapchainExtent = extent;
        }

        void CreateImageViews() {
            if (swapchainImageViews != null) {
                foreach (var iv in swapchainImageViews) iv.Dispose();
            }

            swapchainImageViews = new List<ImageView>();
            foreach (var image in swapchainImages) {
                var info = new ImageViewCreateInfo();
                info.image = image;
                info.viewType = VkImageViewType._2D;
                info.format = swapchainImageFormat;
                info.components.r = VkComponentSwizzle.Identity;
                info.components.g = VkComponentSwizzle.Identity;
                info.components.b = VkComponentSwizzle.Identity;
                info.components.a = VkComponentSwizzle.Identity;
                info.subresourceRange.aspectMask = VkImageAspectFlags.ColorBit;
                info.subresourceRange.baseMipLevel = 0;
                info.subresourceRange.levelCount = 1;
                info.subresourceRange.baseArrayLayer = 0;
                info.subresourceRange.layerCount = 1;

                swapchainImageViews.Add(new ImageView(device, info));
            }
        }

        void CreateRenderPass() {
            var colorAttachment = new AttachmentDescription();
            colorAttachment.format = swapchainImageFormat;
            colorAttachment.samples = VkSampleCountFlags._1_Bit;
            colorAttachment.loadOp = VkAttachmentLoadOp.Clear;
            colorAttachment.storeOp = VkAttachmentStoreOp.Store;
            colorAttachment.stencilLoadOp = VkAttachmentLoadOp.DontCare;
            colorAttachment.stencilStoreOp = VkAttachmentStoreOp.DontCare;
            colorAttachment.initialLayout = VkImageLayout.Undefined;
            colorAttachment.finalLayout = VkImageLayout.PresentSrcKhr;

            var colorAttachmentRef = new AttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VkImageLayout.ColorAttachmentOptimal;

            var subpass = new SubpassDescription();
            subpass.pipelineBindPoint = VkPipelineBindPoint.Graphics;
            subpass.colorAttachments = new List<AttachmentReference> { colorAttachmentRef };

            var dependency = new SubpassDependency();
            dependency.srcSubpass = uint.MaxValue;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VkPipelineStageFlags.BottomOfPipeBit;
            dependency.srcAccessMask = VkAccessFlags.MemoryReadBit;
            dependency.dstStageMask = VkPipelineStageFlags.ColorAttachmentOutputBit;
            dependency.dstAccessMask = VkAccessFlags.ColorAttachmentReadBit
                                    | VkAccessFlags.ColorAttachmentWriteBit;

            var info = new RenderPassCreateInfo();
            info.attachments = new List<AttachmentDescription> { colorAttachment };
            info.subpasses = new List<SubpassDescription> { subpass };
            info.dependencies = new List<SubpassDependency> { dependency };

            renderPass?.Dispose();
            renderPass = new RenderPass(device, info);
        }

        void CreateDescriptorSetLayout() {
            var uboLayoutBinding = new VkDescriptorSetLayoutBinding();
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VkDescriptorType.UniformBuffer;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.stageFlags = VkShaderStageFlags.VertexBit;

            var info = new DescriptorSetLayoutCreateInfo();
            info.bindings = new List<VkDescriptorSetLayoutBinding> { uboLayoutBinding };

            descriptorSetLayout = new DescriptorSetLayout(device, info);
        }

        public ShaderModule CreateShaderModule(byte[] code) {
            var info = new ShaderModuleCreateInfo();
            info.data = code;
            return new ShaderModule(device, info);
        }

        void CreateGraphicsPipeline() {
            var vert = CreateShaderModule(File.ReadAllBytes("vert.spv"));
            var frag = CreateShaderModule(File.ReadAllBytes("frag.spv"));

            var vertInfo = new PipelineShaderStageCreateInfo();
            vertInfo.stage = VkShaderStageFlags.VertexBit;
            vertInfo.module = vert;
            vertInfo.name = "main";

            var fragInfo = new PipelineShaderStageCreateInfo();
            fragInfo.stage = VkShaderStageFlags.FragmentBit;
            fragInfo.module = frag;
            fragInfo.name = "main";

            var shaderStages = new List<PipelineShaderStageCreateInfo> { vertInfo, fragInfo };

            var vertexInputInfo = new PipelineVertexInputStateCreateInfo();
            vertexInputInfo.vertexBindingDescriptions = new List<VkVertexInputBindingDescription> { Vertex.GetBindingDescription() };
            vertexInputInfo.vertexAttributeDescriptions = Vertex.GetAttributeDescriptions();

            var inputAssembly = new PipelineInputAssemblyStateCreateInfo();
            inputAssembly.topology = VkPrimitiveTopology.TriangleList;

            var viewport = new VkViewport();
            viewport.width = swapchainExtent.width;
            viewport.height = swapchainExtent.height;
            viewport.minDepth = 0f;
            viewport.maxDepth = 1f;

            var scissor = new VkRect2D();
            scissor.extent = swapchainExtent;

            var viewportState = new PipelineViewportStateCreateInfo();
            viewportState.viewports = new List<VkViewport> { viewport };
            viewportState.scissors = new List<VkRect2D> { scissor };

            var rasterizer = new PipelineRasterizationStateCreateInfo();
            rasterizer.polygonMode = VkPolygonMode.Fill;
            rasterizer.lineWidth = 1f;
            rasterizer.cullMode = VkCullModeFlags.BackBit;
            rasterizer.frontFace = VkFrontFace.CounterClockwise;

            var multisampling = new PipelineMultisampleStateCreateInfo();
            multisampling.rasterizationSamples = VkSampleCountFlags._1_Bit;
            multisampling.minSampleShading = 1f;

            var colorBlendAttachment = new PipelineColorBlendAttachmentState();
            colorBlendAttachment.colorWriteMask = VkColorComponentFlags.RBit
                                                | VkColorComponentFlags.GBit
                                                | VkColorComponentFlags.BBit
                                                | VkColorComponentFlags.ABit;
            colorBlendAttachment.srcColorBlendFactor = VkBlendFactor.One;
            colorBlendAttachment.dstColorBlendFactor = VkBlendFactor.Zero;
            colorBlendAttachment.colorBlendOp = VkBlendOp.Add;
            colorBlendAttachment.srcAlphaBlendFactor = VkBlendFactor.One;
            colorBlendAttachment.dstAlphaBlendFactor = VkBlendFactor.Zero;
            colorBlendAttachment.alphaBlendOp = VkBlendOp.Add;

            var colorBlending = new PipelineColorBlendStateCreateInfo();
            colorBlending.logicOp = VkLogicOp.Copy;
            colorBlending.attachments = new List<PipelineColorBlendAttachmentState> { colorBlendAttachment };

            var pipelineLayoutInfo = new PipelineLayoutCreateInfo();
            pipelineLayoutInfo.setLayouts = new List<DescriptorSetLayout> { descriptorSetLayout };

            pipelineLayout?.Dispose();

            pipelineLayout = new PipelineLayout(device, pipelineLayoutInfo);

            var info = new GraphicsPipelineCreateInfo();
            info.stages = shaderStages;
            info.vertexInputState = vertexInputInfo;
            info.inputAssemblyState = inputAssembly;
            info.viewportState = viewportState;
            info.rasterizationState = rasterizer;
            info.multisampleState = multisampling;
            info.colorBlendState = colorBlending;
            info.layout = pipelineLayout;
            info.renderPass = renderPass;
            info.subpass = 0;
            info.basePipelineHandle = null;
            info.basePipelineIndex = -1;

            pipeline?.Dispose();

            pipeline = new GraphicsPipeline(device, info, null);

            vert.Dispose();
            frag.Dispose();
        }

        void CreateFramebuffers() {
            if (swapchainFramebuffers != null) {
                foreach (var fb in swapchainFramebuffers) fb.Dispose();
            }

            swapchainFramebuffers = new List<Framebuffer>(swapchainImageViews.Count);

            for (int i = 0; i < swapchainImageViews.Count; i++) {
                var attachments = new List<ImageView> { swapchainImageViews[i] };

                var info = new FramebufferCreateInfo();
                info.renderPass = renderPass;
                info.attachments = attachments;
                info.width = swapchainExtent.width;
                info.height = swapchainExtent.height;
                info.layers = 1;

                swapchainFramebuffers.Add(new Framebuffer(device, info));
            }
        }

        void CreateCommandPool() {
            var info = new CommandPoolCreateInfo();
            info.queueFamilyIndex = graphicsIndex;

            commandPool = new CommandPool(device, info);
        }

        void CreateBuffer(ulong size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, out Buffer buffer, out DeviceMemory memory) {
            var info = new BufferCreateInfo();
            info.size = size;
            info.usage = usage;
            info.sharingMode = VkSharingMode.Exclusive;

            buffer = new Buffer(device, info);

            var allocInfo = new MemoryAllocateInfo();
            allocInfo.allocationSize = buffer.Requirements.size;
            allocInfo.memoryTypeIndex = FindMemoryType(buffer.Requirements.memoryTypeBits, properties);

            memory = new DeviceMemory(device, allocInfo);
            buffer.Bind(memory, 0);
        }

        void CreateVertexBuffer() {
            ulong bufferSize = (ulong)Interop.SizeOf(vertices);
            Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;
            CreateBuffer(bufferSize,
                VkBufferUsageFlags.TransferSrcBit,
                VkMemoryPropertyFlags.HostVisibleBit
                | VkMemoryPropertyFlags.HostCoherentBit,
                out stagingBuffer,
                out stagingBufferMemory);

            var data = stagingBufferMemory.Map(0, bufferSize);
            Interop.Copy(vertices, data);
            stagingBufferMemory.Unmap();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.TransferDstBit
                | VkBufferUsageFlags.VertexBufferBit,
                VkMemoryPropertyFlags.DeviceLocalBit,
                out vertexBuffer,
                out vertexBufferMemory);

            CopyBuffer(stagingBuffer, vertexBuffer, bufferSize);

            stagingBuffer.Dispose();
            stagingBufferMemory.Dispose();
        }

        void CreateIndexBuffer() {
            ulong bufferSize = (ulong)Interop.SizeOf(indices);
            Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;
            CreateBuffer(bufferSize,
                VkBufferUsageFlags.TransferSrcBit,
                VkMemoryPropertyFlags.HostVisibleBit
                | VkMemoryPropertyFlags.HostCoherentBit,
                out stagingBuffer,
                out stagingBufferMemory);

            var data = stagingBufferMemory.Map(0, bufferSize);
            Interop.Copy(indices, data);
            stagingBufferMemory.Unmap();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.TransferDstBit
                | VkBufferUsageFlags.IndexBufferBit,
                VkMemoryPropertyFlags.DeviceLocalBit,
                out indexBuffer,
                out indexBufferMemory);

            CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

            stagingBuffer.Dispose();
            stagingBufferMemory.Dispose();
        }

        void CreateUniformBuffer() {
            ulong bufferSize = (ulong)Interop.SizeOf<UniformBufferObject>();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.TransferDstBit
                | VkBufferUsageFlags.UniformBufferBit,
                VkMemoryPropertyFlags.HostVisibleBit
                | VkMemoryPropertyFlags.HostCoherentBit,
                out uniformBuffer,
                out uniformBufferMemory);
        }

        void CopyBuffer(Buffer src, Buffer dst, ulong size) {
            var info = new CommandBufferAllocateInfo();
            info.level = VkCommandBufferLevel.Primary;
            info.commandBufferCount = 1;

            var buffers = commandPool.Allocate(info);
            var buffer = buffers[0];

            var beginInfo = new CommandBufferBeginInfo();
            beginInfo.flags = VkCommandBufferUsageFlags.OneTimeSubmitBit;

            buffer.Begin(beginInfo);

            VkBufferCopy region = new VkBufferCopy();
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;

            buffer.CopyBuffer(src, dst, new VkBufferCopy[] { region });
            buffer.End();

            var submitInfo = new SubmitInfo();
            submitInfo.commandBuffers = new List<CommandBuffer> { buffer };

            graphicsQueue.Submit(new List<SubmitInfo> { submitInfo }, null);
            graphicsQueue.WaitIdle();

            commandPool.Free(buffers);
        }

        uint FindMemoryType(uint filter, VkMemoryPropertyFlags flags) {
            var props = physicalDevice.MemoryProperties;

            for (int i = 0; i < props.memoryTypeCount; i++) {
                if ((filter & (1 << i)) != 0 && (props.GetMemoryTypes(i).propertyFlags & flags) == flags) {
                    return (uint)i;
                }
            }

            throw new Exception("Failed to find suitable memory type");
        }

        void CreateDescriptorPool() {
            var poolSize = new VkDescriptorPoolSize();
            poolSize.type = VkDescriptorType.UniformBuffer;
            poolSize.descriptorCount = 1;

            var info = new DescriptorPoolCreateInfo();
            info.poolSizes = new List<VkDescriptorPoolSize> { poolSize };
            info.maxSets = 1;

            descriptorPool = new DescriptorPool(device, info);
        }

        void CreateDescriptorSet() {
            var layouts = new List<DescriptorSetLayout> { descriptorSetLayout };
            var info = new DescriptorSetAllocateInfo();
            info.setLayouts = layouts;

            descriptorSet = descriptorPool.Allocate(info)[0];

            var bufferInfo = new DescriptorBufferInfo();
            bufferInfo.buffer = uniformBuffer;
            bufferInfo.offset = 0;
            bufferInfo.range = (ulong)Interop.SizeOf<UniformBufferObject>();

            var descriptorWrites = new List<WriteDescriptorSet>();
            descriptorWrites.Add(new WriteDescriptorSet());
            descriptorWrites[0].dstSet = descriptorSet;
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VkDescriptorType.UniformBuffer;
            descriptorWrites[0].bufferInfo = new List<DescriptorBufferInfo> { bufferInfo };

            descriptorSet.Update(descriptorWrites, null);
        }

        void CreateCommandBuffers() {
            if (commandBuffers != null) {
                commandPool.Free(commandBuffers);
            }

            var info = new CommandBufferAllocateInfo();
            info.level = VkCommandBufferLevel.Primary;
            info.commandBufferCount = (uint)swapchainFramebuffers.Count;

            commandBuffers = new List<CommandBuffer>(commandPool.Allocate(info));

            for (int i = 0; i < commandBuffers.Count; i++) {
                var buffer = commandBuffers[i];
                var beginInfo = new CommandBufferBeginInfo();
                beginInfo.flags = VkCommandBufferUsageFlags.SimultaneousUseBit;

                buffer.Begin(beginInfo);

                var renderPassInfo = new RenderPassBeginInfo();
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapchainFramebuffers[i];
                renderPassInfo.renderArea.extent = swapchainExtent;

                VkClearValue clearColor = new VkClearValue();
                clearColor.color.float32_0 = 0;
                clearColor.color.float32_1 = 0;
                clearColor.color.float32_2 = 0;
                clearColor.color.float32_3 = 1f;

                renderPassInfo.clearValues = new List<VkClearValue> { clearColor };

                buffer.BeginRenderPass(renderPassInfo, VkSubpassContents.Inline);
                buffer.BindPipeline(VkPipelineBindPoint.Graphics, pipeline);
                buffer.BindVertexBuffers(0, new Buffer[] { vertexBuffer }, new ulong[] { 0 });
                buffer.BindIndexBuffer(indexBuffer, 0, VkIndexType.UINT32);
                buffer.BindDescriptorSets(VkPipelineBindPoint.Graphics, pipelineLayout, 0, new DescriptorSet[] { descriptorSet }, null);
                buffer.DrawIndexed((uint)indices.Length, 1, 0, 0, 0);
                buffer.EndRenderPass();
                buffer.End();
            }
        }

        void CreateSemaphores() {
            imageAvailableSemaphore = new Semaphore(device);
            renderFinishedSemaphore = new Semaphore(device);
        }
    }

    struct SwapchainSupport {
        public VkSurfaceCapabilitiesKHR cap;
        public List<VkSurfaceFormatKHR> formats;
        public List<VkPresentModeKHR> modes;

        public SwapchainSupport(VkSurfaceCapabilitiesKHR cap, List<VkSurfaceFormatKHR> formats, List<VkPresentModeKHR> modes) {
            this.cap = cap;
            this.formats = formats;
            this.modes = modes;
        }
    }
}
