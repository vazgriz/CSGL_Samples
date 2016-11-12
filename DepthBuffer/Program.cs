using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Diagnostics;
using System.Drawing;

using CSGL;
using CSGL.GLFW;
using CSGL.Vulkan;

using Buffer = CSGL.Vulkan.Buffer;
using Image = CSGL.Vulkan.Image;

namespace Samples {
    public struct Vertex {
        public Vector3 position;
        public Vector3 color;
        public Vector2 texCoord;

        public Vertex(Vector3 position, Vector3 color, Vector2 texCoord) {
            this.position = position;
            this.color = color;
            this.texCoord = texCoord;
        }

        public static VkVertexInputBindingDescription GetBindingDescription() {
            var result = new VkVertexInputBindingDescription();
            result.binding = 0;
            result.stride = (uint)Interop.SizeOf<Vertex>();
            result.inputRate = VkVertexInputRate.VertexInputRateVertex;

            return result;
        }

        public static VkVertexInputAttributeDescription[] GetAttributeDescriptions() {
            Vertex v = new Vertex();
            var result = new VkVertexInputAttributeDescription[3];

            result[0].binding = 0;
            result[0].location = 0;
            result[0].format = VkFormat.FormatR32g32b32Sfloat;
            result[0].offset = (uint)Interop.Offset(ref v, ref v.position);
            
            result[1].binding = 0;
            result[1].location = 1;
            result[1].format = VkFormat.FormatR32g32b32Sfloat;
            result[1].offset = (uint)Interop.Offset(ref v, ref v.color);

            result[2].binding = 0;
            result[2].location = 2;
            result[2].format = VkFormat.FormatR32g32Sfloat;
            result[2].offset = (uint)Interop.Offset(ref v, ref v.texCoord);

            return result;
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

        string[] layers = {
            "VK_LAYER_LUNARG_standard_validation",
            //"VK_LAYER_LUNARG_api_dump"
        };

        string[] deviceExtensions = {
            "VK_KHR_swapchain"
        };

        string image = "lenna.png";

        Vertex[] vertices = {
            new Vertex(new Vector3(-1f, -1f, 0), new Vector3(1, 0, 0), new Vector2(0, 0)),
            new Vertex(new Vector3(1f, -1f, 0),  new Vector3(0, 1, 0), new Vector2(1, 0)),
            new Vertex(new Vector3(1f, 1f, 0),   new Vector3(0, 0, 1), new Vector2(1, 1)),
            new Vertex(new Vector3(-1f, 1f, 0),  new Vector3(1, 1, 1), new Vector2(0, 1)),

            new Vertex(new Vector3(-1f, -1f, -0.5f), new Vector3(1, 0, 0), new Vector2(0, 0)),
            new Vertex(new Vector3(1f, -1f, -0.5f),  new Vector3(0, 1, 0), new Vector2(1, 0)),
            new Vertex(new Vector3(1f, 1f, -0.5f),   new Vector3(0, 0, 1), new Vector2(1, 1)),
            new Vertex(new Vector3(-1f, 1f, -0.5f),  new Vector3(1, 1, 1), new Vector2(0, 1)),
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0,
            4, 5, 6, 6, 7, 4
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
        Image depthImage;
        DeviceMemory depthImageMemory;
        ImageView depthImageView;
        Image textureImage;
        DeviceMemory textureImageMemory;
        ImageView textureImageView;
        Sampler textureSampler;
        Buffer vertexBuffer;
        DeviceMemory vertexBufferMemory;
        Buffer indexBuffer;
        DeviceMemory indexBufferMemory;
        Buffer uniformStagingBuffer;
        DeviceMemory uniformStagingBufferMemory;
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
            Vulkan.Init();
            CreateWindow();
            CreateInstance();
            PickPhysicalDevice();
            CreateSurface();
            PickQueues();
            CreateDevice();
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateDescriptorSetLayout();
            CreateGraphicsPipeline();
            CreateCommandPool();
            CreateDepthResources();
            CreateFramebuffers();
            CreateTextureImage();
            CreateTextureImageView();
            CreateTextureSampler();
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
            uniformStagingBufferMemory.Dispose();
            uniformStagingBuffer.Dispose();
            indexBufferMemory.Dispose();
            indexBuffer.Dispose();
            vertexBufferMemory.Dispose();
            vertexBuffer.Dispose();
            textureSampler.Dispose();
            textureImageView.Dispose();
            textureImageMemory.Dispose();
            textureImage.Dispose();
            depthImageView.Dispose();
            depthImageMemory.Dispose();
            depthImage.Dispose();
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
            instance.Dispose();
            GLFW.DestroyWindow(window);
            GLFW.Terminate();
        }

        void MainLoop() {
            var waitSemaphores = new Semaphore[] { imageAvailableSemaphore };
            var waitStages = new VkPipelineStageFlags[] { VkPipelineStageFlags.PipelineStageColorAttachmentOutputBit };
            var signalSemaphores = new Semaphore[] { renderFinishedSemaphore };
            var swapchains = new Swapchain[] { swapchain };

            var commandBuffer = new CommandBuffer[1];
            var index = new uint[1];

            var submitInfo = new SubmitInfo();
            submitInfo.waitSemaphores = waitSemaphores;
            submitInfo.waitDstStageMask = waitStages;
            submitInfo.commandBuffers = commandBuffer;
            submitInfo.signalSemaphores = signalSemaphores;

            var presentInfo = new PresentInfo();
            presentInfo.waitSemaphores = signalSemaphores;
            presentInfo.swapchains = swapchains;
            presentInfo.imageIndices = index;

            var submitInfos = new SubmitInfo[] { submitInfo };

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
                var result = swapchain.AcquireNextImage(ulong.MaxValue, imageAvailableSemaphore, out imageIndex);

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

            var data = uniformStagingBufferMemory.Map(0, size, VkMemoryMapFlags.None);
            Interop.Copy(new UniformBufferObject[] { ubo }, data);
            uniformStagingBufferMemory.Unmap();

            CopyBuffer(uniformStagingBuffer, uniformBuffer, size);
        }

        void RecreateSwapchain() {
            device.WaitIdle();
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicsPipeline();
            CreateDepthResources();
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
            window = GLFW.CreateWindow(width, height, "Vulkan Test", MonitorPtr.Null, WindowPtr.Null);
        }

        void CreateInstance() {
            var extensions = GLFW.GetRequiredInstanceExceptions();

            var appInfo = new ApplicationInfo(
                new VkVersion(1, 0, 0),
                new VkVersion(1, 0, 0),
                new VkVersion(1, 0, 0),
                "Vulkan Test",
                null
            );

            var info = new InstanceCreateInfo(appInfo, extensions, layers);
            instance = new Instance(info);
        }

        void PickPhysicalDevice() {
            physicalDevice = instance.PhysicalDevices[0];
        }

        void CreateSurface() {
            surface = new Surface(physicalDevice, window);
        }

        void PickQueues() {
            int g = -1;
            int p = -1;

            for (int i = 0; i < physicalDevice.QueueFamilies.Count; i++) {
                var family = physicalDevice.QueueFamilies[i];
                if ((family.Flags & VkQueueFlags.QueueGraphicsBit) != 0) {
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
            float[] priorities = new float[] { 1f };
            DeviceQueueCreateInfo[] queueInfos = new DeviceQueueCreateInfo[uniqueIndices.Count];

            int i = 0;
            foreach (var ind in uniqueIndices) {
                var queueInfo = new DeviceQueueCreateInfo(ind, 1, priorities);
                queueInfos[i] = queueInfo;
                i++;
            }

            var info = new DeviceCreateInfo(deviceExtensions, queueInfos, features);
            device = new Device(physicalDevice, info);

            graphicsQueue = device.GetQueue(graphicsIndex, 0);
            presentQueue = device.GetQueue(presentIndex, 0);
        }

        SwapchainSupport GetSwapchainSupport(PhysicalDevice physicalDevice) {
            var cap = surface.Capabilities;
            var formats = surface.Formats;
            var modes = surface.PresentModes;

            return new SwapchainSupport(cap, formats, modes);
        }

        VkSurfaceFormatKHR ChooseSwapSurfaceFormat(List<VkSurfaceFormatKHR> formats) {
            if (formats.Count == 1 && formats[0].format == VkFormat.FormatUndefined) {
                var result = new VkSurfaceFormatKHR();
                result.format = VkFormat.FormatB8g8r8a8Unorm;
                result.colorSpace = VkColorSpaceKHR.ColorSpaceSrgbNonlinearKhr;
                return result;
            }

            foreach (var f in formats) {
                if (f.format == VkFormat.FormatB8g8r8a8Unorm && f.colorSpace == VkColorSpaceKHR.ColorSpaceSrgbNonlinearKhr) {
                    return f;
                }
            }

            return formats[0];
        }

        VkPresentModeKHR ChooseSwapPresentMode(List<VkPresentModeKHR> modes) {
            foreach (var m in modes) {
                if (m == VkPresentModeKHR.PresentModeMailboxKhr) {
                    return m;
                }
            }

            return VkPresentModeKHR.PresentModeFifoKhr;
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
            var info = new SwapchainCreateInfo(surface, oldSwapchain);
            info.minImageCount = imageCount;
            info.imageFormat = surfaceFormat.format;
            info.imageColorSpace = surfaceFormat.colorSpace;
            info.imageExtent = extent;
            info.imageArrayLayers = 1;
            info.imageUsage = VkImageUsageFlags.ImageUsageColorAttachmentBit;

            var queueFamilyIndices = new uint[] { graphicsIndex, presentIndex };

            if (graphicsIndex != presentIndex) {
                info.imageSharingMode = VkSharingMode.SharingModeConcurrent;
                info.queueFamilyIndices = queueFamilyIndices;
            } else {
                info.imageSharingMode = VkSharingMode.SharingModeExclusive;
            }

            info.preTransform = cap.currentTransform;
            info.compositeAlpha = VkCompositeAlphaFlagsKHR.CompositeAlphaOpaqueBitKhr;
            info.presentMode = mode;
            info.clipped = true;

            swapchain = new Swapchain(device, info);
            oldSwapchain?.Dispose();

            swapchainImages = swapchain.Images;

            swapchainImageFormat = surfaceFormat.format;
            swapchainExtent = extent;
        }

        void CreateImageView(Image image, VkFormat format, VkImageAspectFlags aspectFlags, ref ImageView imageView) {
            var info = new ImageViewCreateInfo(image);
            info.viewType = VkImageViewType.ImageViewType2d;
            info.format = format;
            info.subresourceRange.aspectMask = aspectFlags;
            info.subresourceRange.baseMipLevel = 0; ;
            info.subresourceRange.levelCount = 1;
            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            imageView?.Dispose();
            imageView = new ImageView(device, info);
        }

        void CreateImageViews() {
            if (swapchainImageViews != null) {
                foreach (var iv in swapchainImageViews) iv.Dispose();
            }

            swapchainImageViews = new List<ImageView>();
            foreach (var image in swapchainImages) {
                ImageView temp = null;
                CreateImageView(image, swapchainImageFormat, VkImageAspectFlags.ImageAspectColorBit, ref temp);
                swapchainImageViews.Add(temp);
            }
        }

        void CreateRenderPass() {
            var colorAttachment = new VkAttachmentDescription();
            colorAttachment.format = swapchainImageFormat;
            colorAttachment.samples = VkSampleCountFlags.SampleCount1Bit;
            colorAttachment.loadOp = VkAttachmentLoadOp.AttachmentLoadOpClear;
            colorAttachment.storeOp = VkAttachmentStoreOp.AttachmentStoreOpStore;
            colorAttachment.stencilLoadOp = VkAttachmentLoadOp.AttachmentLoadOpDontCare;
            colorAttachment.stencilStoreOp = VkAttachmentStoreOp.AttachmentStoreOpDontCare;
            colorAttachment.initialLayout = VkImageLayout.ImageLayoutUndefined;
            colorAttachment.finalLayout = VkImageLayout.ImageLayoutPresentSrcKhr;

            var depthAttachment = new VkAttachmentDescription();
            depthAttachment.format = FindDepthFormat();
            depthAttachment.samples = VkSampleCountFlags.SampleCount1Bit;
            depthAttachment.loadOp = VkAttachmentLoadOp.AttachmentLoadOpClear;
            depthAttachment.storeOp = VkAttachmentStoreOp.AttachmentStoreOpDontCare;
            depthAttachment.stencilLoadOp = VkAttachmentLoadOp.AttachmentLoadOpDontCare;
            depthAttachment.stencilStoreOp = VkAttachmentStoreOp.AttachmentStoreOpDontCare;
            depthAttachment.initialLayout = VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal;
            depthAttachment.finalLayout = VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal;

            var colorAttachmentRef = new VkAttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VkImageLayout.ImageLayoutColorAttachmentOptimal;

            var depthAttachmentRef = new VkAttachmentReference();
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal;

            var subpass = new SubpassDescription();
            subpass.PipelineBindPoint = VkPipelineBindPoint.PipelineBindPointGraphics;
            subpass.ColorAttachments = new VkAttachmentReference[] { colorAttachmentRef };
            subpass.DepthStencilAttachment = depthAttachmentRef;

            var dependency = new VkSubpassDependency();
            dependency.srcSubpass = uint.MaxValue;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VkPipelineStageFlags.PipelineStageBottomOfPipeBit;
            dependency.srcAccessMask = VkAccessFlags.AccessMemoryReadBit;
            dependency.dstStageMask = VkPipelineStageFlags.PipelineStageColorAttachmentOutputBit;
            dependency.dstAccessMask = VkAccessFlags.AccessColorAttachmentReadBit
                                    | VkAccessFlags.AccessColorAttachmentWriteBit;

            var info = new RenderPassCreateInfo();
            info.Attachments = new VkAttachmentDescription[] { colorAttachment, depthAttachment };
            info.Subpasses = new SubpassDescription[] { subpass };
            info.Dependencies = new VkSubpassDependency[] { dependency };

            renderPass?.Dispose();
            renderPass = new RenderPass(device, info);
        }

        void CreateDescriptorSetLayout() {
            var uboLayoutBinding = new VkDescriptorSetLayoutBinding();
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VkDescriptorType.DescriptorTypeUniformBuffer;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.stageFlags = VkShaderStageFlags.ShaderStageVertexBit;

            var samplerLayoutBinding = new VkDescriptorSetLayoutBinding();
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VkDescriptorType.DescriptorTypeCombinedImageSampler;
            samplerLayoutBinding.stageFlags = VkShaderStageFlags.ShaderStageFragmentBit;

            var info = new DescriptorSetLayoutCreateInfo();
            info.bindings = new VkDescriptorSetLayoutBinding[] { uboLayoutBinding, samplerLayoutBinding };

            descriptorSetLayout = new DescriptorSetLayout(device, info);
        }

        public ShaderModule CreateShaderModule(byte[] code) {
            var info = new ShaderModuleCreateInfo(code);
            return new ShaderModule(device, info);
        }

        void CreateGraphicsPipeline() {
            var vert = CreateShaderModule(File.ReadAllBytes("vert.spv"));
            var frag = CreateShaderModule(File.ReadAllBytes("frag.spv"));

            var vertInfo = new PipelineShaderStageCreateInfo();
            vertInfo.stage = VkShaderStageFlags.ShaderStageVertexBit;
            vertInfo.module = vert;
            vertInfo.name = "main";

            var fragInfo = new PipelineShaderStageCreateInfo();
            fragInfo.stage = VkShaderStageFlags.ShaderStageFragmentBit;
            fragInfo.module = frag;
            fragInfo.name = "main";

            var shaderStages = new PipelineShaderStageCreateInfo[] { vertInfo, fragInfo };

            var vertexInputInfo = new PipelineVertexInputStateCreateInfo();
            vertexInputInfo.vertexBindingDescriptions = new VkVertexInputBindingDescription[] { Vertex.GetBindingDescription() };
            vertexInputInfo.vertexAttributeDescriptions = Vertex.GetAttributeDescriptions();

            var inputAssembly = new PipelineInputAssemblyStateCreateInfo();
            inputAssembly.topology = VkPrimitiveTopology.PrimitiveTopologyTriangleList;

            var viewport = new VkViewport();
            viewport.width = swapchainExtent.width;
            viewport.height = swapchainExtent.height;
            viewport.minDepth = 0f;
            viewport.maxDepth = 1f;

            var scissor = new VkRect2D();
            scissor.extent = swapchainExtent;

            var viewportState = new PipelineViewportStateCreateInfo();
            viewportState.viewports = new VkViewport[] { viewport };
            viewportState.scissors = new VkRect2D[] { scissor };

            var rasterizer = new PipelineRasterizationStateCreateInfo();
            rasterizer.polygonMode = VkPolygonMode.PolygonModeFill;
            rasterizer.lineWidth = 1f;
            rasterizer.cullMode = VkCullModeFlags.CullModeBackBit;
            rasterizer.frontFace = VkFrontFace.FrontFaceCounterClockwise;

            var multisampling = new PipelineMultisampleStateCreateInfo();
            multisampling.rasterizationSamples = VkSampleCountFlags.SampleCount1Bit;
            multisampling.minSampleShading = 1f;

            var colorBlendAttachment = new PipelineColorBlendAttachmentState();
            colorBlendAttachment.colorWriteMask = VkColorComponentFlags.ColorComponentRBit
                                                | VkColorComponentFlags.ColorComponentGBit
                                                | VkColorComponentFlags.ColorComponentBBit
                                                | VkColorComponentFlags.ColorComponentABit;
            colorBlendAttachment.srcColorBlendFactor = VkBlendFactor.BlendFactorOne;
            colorBlendAttachment.dstColorBlendFactor = VkBlendFactor.BlendFactorZero;
            colorBlendAttachment.colorBlendOp = VkBlendOp.BlendOpAdd;
            colorBlendAttachment.srcAlphaBlendFactor = VkBlendFactor.BlendFactorOne;
            colorBlendAttachment.dstAlphaBlendFactor = VkBlendFactor.BlendFactorZero;
            colorBlendAttachment.alphaBlendOp = VkBlendOp.BlendOpAdd;

            var colorBlending = new PipelineColorBlendStateCreateInfo();
            colorBlending.logicOp = VkLogicOp.LogicOpCopy;
            colorBlending.attachments = new PipelineColorBlendAttachmentState[] { colorBlendAttachment };

            var depthStencil = new PipelineDepthStencilStateCreateInfo();
            depthStencil.depthTestEnable = true;
            depthStencil.depthWriteEnable = true;
            depthStencil.depthCompareOp = VkCompareOp.CompareOpLess;
            depthStencil.depthBoundsTestEnable = false;
            depthStencil.minDepthBounds = 0;
            depthStencil.maxDepthBounds = 1;
            depthStencil.stencilTestEnable = false;

            var pipelineLayoutInfo = new PipelineLayoutCreateInfo();
            pipelineLayoutInfo.setLayouts = new DescriptorSetLayout[] { descriptorSetLayout };

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
            info.depthStencilState = depthStencil;
            info.layout = pipelineLayout;
            info.renderPass = renderPass;
            info.subpass = 0;
            info.basePipeline = null;
            info.basePipelineIndex = -1;

            pipeline?.Dispose();

            pipeline = new Pipeline(device, info, null);

            vert.Dispose();
            frag.Dispose();
        }

        void CreateFramebuffers() {
            if (swapchainFramebuffers != null) {
                foreach (var fb in swapchainFramebuffers) fb.Dispose();
            }

            swapchainFramebuffers = new List<Framebuffer>(swapchainImageViews.Count);

            for (int i = 0; i < swapchainImageViews.Count; i++) {
                var attachments = new ImageView[] { swapchainImageViews[i], depthImageView };

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
            info.QueueFamilyIndex = graphicsIndex;

            commandPool = new CommandPool(device, info);
        }

        void CreateDepthResources() {
            depthImage?.Dispose();
            depthImageMemory?.Dispose();
            depthImageView?.Dispose();

            VkFormat depthFormat = FindDepthFormat();
            CreateImage(swapchainExtent.width, swapchainExtent.height, depthFormat,
                VkImageTiling.ImageTilingOptimal, VkImageUsageFlags.ImageUsageDepthStencilAttachmentBit,
                VkMemoryPropertyFlags.MemoryPropertyDeviceLocalBit,
                out depthImage, out depthImageMemory);

            CreateImageView(depthImage, depthFormat, VkImageAspectFlags.ImageAspectDepthBit, ref depthImageView);

            TransitionImageLayout(depthImage, depthFormat, VkImageLayout.ImageLayoutUndefined, VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal);
        }

        VkFormat FindSupportedFormat(List<VkFormat> candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
            foreach (var format in candidates) {
                var props = physicalDevice.GetFormatProperties(format);

                if (tiling == VkImageTiling.ImageTilingLinear && (props.linearTilingFeatures & features) == features) {
                    return format;
                } else if (tiling == VkImageTiling.ImageTilingOptimal && (props.optimalTilingFeatures & features) == features) {
                    return format;
                }
            }

            throw new Exception("Failed to find supported format");
        }

        VkFormat FindDepthFormat() {
            return FindSupportedFormat(
                new List<VkFormat>() { VkFormat.FormatD32Sfloat, VkFormat.FormatD32SfloatS8Uint, VkFormat.FormatD24UnormS8Uint },
                VkImageTiling.ImageTilingOptimal,
                VkFormatFeatureFlags.FormatFeatureDepthStencilAttachmentBit);
        }

        bool HasStencilComponent(VkFormat format) {
            return format == VkFormat.FormatD32SfloatS8Uint || format == VkFormat.FormatD24UnormS8Uint;
        }

        void CreateBuffer(ulong size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, out Buffer buffer, out DeviceMemory memory) {
            var info = new BufferCreateInfo();
            info.size = size;
            info.usage = usage;
            info.sharingMode = VkSharingMode.SharingModeExclusive;

            buffer = new Buffer(device, info);

            var allocInfo = new MemoryAllocateInfo();
            allocInfo.allocationSize = buffer.Requirements.size;
            allocInfo.memoryTypeIndex = FindMemoryType(buffer.Requirements.memoryTypeBits, properties);

            memory = new DeviceMemory(device, allocInfo);
            buffer.Bind(memory, 0);
        }

        void CreateTextureImage() {
            using (var bitmap = new Bitmap(image)) {
                var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                    System.Drawing.Imaging.ImageLockMode.ReadOnly,
                    System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                var pixels = bitmapData.Scan0;

                ulong imageSize = (ulong)(bitmap.Width * bitmap.Height * 4);

                //bitmap loads data as BGRA
                //so we must swap the B and R values to get RGBA

                unsafe
                {
                    int length = bitmapData.Width * bitmapData.Height * 4;
                    byte* ptr = (byte*)bitmapData.Scan0;

                    for (int i = 0; i < length; i += 4) {
                        byte* r = &ptr[i];
                        byte* b = &ptr[i + 2];

                        byte temp = *r;
                        *r = *b;
                        *b = temp;
                    }
                }

                Image stagingImage;
                DeviceMemory stagingImageMemory;

                CreateImage((uint)bitmap.Width, (uint)bitmap.Height,
                    VkFormat.FormatR8g8b8a8Unorm, VkImageTiling.ImageTilingLinear,
                    VkImageUsageFlags.ImageUsageTransferSrcBit,
                    VkMemoryPropertyFlags.MemoryPropertyHostCoherentBit
                    | VkMemoryPropertyFlags.MemoryPropertyHostVisibleBit,
                    out stagingImage, out stagingImageMemory);

                var data = stagingImageMemory.Map(0, imageSize, VkMemoryMapFlags.None);
                Interop.Copy(pixels, data, (long)imageSize);
                stagingImageMemory.Unmap();

                CreateImage((uint)bitmap.Width, (uint)bitmap.Height,
                    VkFormat.FormatR8g8b8a8Unorm,
                    VkImageTiling.ImageTilingLinear,
                    VkImageUsageFlags.ImageUsageTransferDstBit | VkImageUsageFlags.ImageUsageSampledBit,
                    VkMemoryPropertyFlags.MemoryPropertyDeviceLocalBit,
                    out textureImage, out textureImageMemory);

                TransitionImageLayout(stagingImage, VkFormat.FormatR8g8b8a8Unorm,
                    VkImageLayout.ImageLayoutPreinitialized, VkImageLayout.ImageLayoutTransferSrcOptimal);
                TransitionImageLayout(textureImage, VkFormat.FormatR8g8b8a8Unorm,
                    VkImageLayout.ImageLayoutPreinitialized, VkImageLayout.ImageLayoutTransferDstOptimal);
                CopyImage(stagingImage, textureImage, (uint)bitmap.Width, (uint)bitmap.Height);

                TransitionImageLayout(textureImage, VkFormat.FormatR8g8b8a8Unorm,
                    VkImageLayout.ImageLayoutTransferDstOptimal, VkImageLayout.ImageLayoutShaderReadOnlyOptimal);

                stagingImage.Dispose();
                stagingImageMemory.Dispose();
                bitmap.UnlockBits(bitmapData);
            }
        }

        void CreateTextureImageView() {
            CreateImageView(textureImage, VkFormat.FormatR8g8b8a8Unorm, VkImageAspectFlags.ImageAspectColorBit, ref textureImageView);
        }

        void CreateTextureSampler() {
            var info = new SamplerCreateInfo();
            info.magFilter = VkFilter.FilterLinear;
            info.minFilter = VkFilter.FilterLinear;
            info.addressModeU = VkSamplerAddressMode.SamplerAddressModeRepeat;
            info.addressModeV = VkSamplerAddressMode.SamplerAddressModeRepeat;
            info.addressModeW = VkSamplerAddressMode.SamplerAddressModeRepeat;
            info.anisotropyEnable = true;
            info.maxAnisotropy = 16;
            info.borderColor = VkBorderColor.BorderColorFloatOpaqueBlack;
            info.unnormalizedCoordinates = false;

            textureSampler = new Sampler(device, info);
        }

        void CreateImage(uint width, uint height,
            VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
            out Image image, out DeviceMemory memory) {

            var info = new ImageCreateInfo();
            info.imageType = VkImageType.ImageType2d;
            info.extent.width = width;
            info.extent.height = height;
            info.extent.depth = 1;
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.format = format;
            info.tiling = tiling;
            info.initialLayout = VkImageLayout.ImageLayoutPreinitialized;
            info.usage = usage;
            info.sharingMode = VkSharingMode.SharingModeExclusive;
            info.samples = VkSampleCountFlags.SampleCount1Bit;

            image = new Image(device, info);

            var req = image.MemoryRequirements;

            var allocInfo = new MemoryAllocateInfo();
            allocInfo.allocationSize = req.size;
            allocInfo.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, properties);

            memory = new DeviceMemory(device, allocInfo);

            image.Bind(memory, 0);
        }

        CommandBuffer BeginSingleTimeCommands() {
            var commandBuffer = commandPool.Allocate(VkCommandBufferLevel.CommandBufferLevelPrimary);

            var beginInfo = new CommandBufferBeginInfo();
            beginInfo.flags = VkCommandBufferUsageFlags.CommandBufferUsageOneTimeSubmitBit;

            commandBuffer.Begin(beginInfo);

            return commandBuffer;
        }

        void EndSingleTimeCommand(CommandBuffer commandBuffer) {
            commandBuffer.End();
            var commands = new CommandBuffer[] { commandBuffer };

            var info = new SubmitInfo();
            info.commandBuffers = commands;

            graphicsQueue.Submit(new SubmitInfo[] { info }, null);
            graphicsQueue.WaitIdle();

            commandPool.Free(commands);
        }

        void TransitionImageLayout(Image image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
            var commandBuffer = BeginSingleTimeCommands();

            var barrier = new ImageMemoryBarrier();
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = uint.MaxValue;    //VK_QUEUE_FAMILY_IGNORED
            barrier.dstQueueFamilyIndex = uint.MaxValue;
            barrier.image = image;

            if (newLayout == VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal) {
                barrier.subresourceRange.aspectMask = VkImageAspectFlags.ImageAspectDepthBit;

                if (HasStencilComponent(format)) {
                    barrier.subresourceRange.aspectMask |= VkImageAspectFlags.ImageAspectStencilBit;
                }
            } else {
                barrier.subresourceRange.aspectMask = VkImageAspectFlags.ImageAspectColorBit;
            }


            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            if (oldLayout == VkImageLayout.ImageLayoutPreinitialized && newLayout == VkImageLayout.ImageLayoutTransferSrcOptimal) {
                barrier.srcAccessMask = VkAccessFlags.AccessHostWriteBit;
                barrier.dstAccessMask = VkAccessFlags.AccessTransferReadBit;
            } else if (oldLayout == VkImageLayout.ImageLayoutPreinitialized && newLayout == VkImageLayout.ImageLayoutTransferDstOptimal) {
                barrier.srcAccessMask = VkAccessFlags.AccessHostWriteBit;
                barrier.dstAccessMask = VkAccessFlags.AccessTransferWriteBit;
            } else if (oldLayout == VkImageLayout.ImageLayoutTransferDstOptimal && newLayout == VkImageLayout.ImageLayoutShaderReadOnlyOptimal) {
                barrier.srcAccessMask = VkAccessFlags.AccessTransferWriteBit;
                barrier.dstAccessMask = VkAccessFlags.AccessShaderReadBit;
            } else if (oldLayout == VkImageLayout.ImageLayoutUndefined && newLayout == VkImageLayout.ImageLayoutDepthStencilAttachmentOptimal) {
                barrier.srcAccessMask = VkAccessFlags.None;
                barrier.dstAccessMask = VkAccessFlags.AccessDepthStencilAttachmentReadBit | VkAccessFlags.AccessDepthStencilAttachmentWriteBit;
            } else {
                throw new Exception("Unsupported layout transition");
            }

            commandBuffer.PipelineBarrier(VkPipelineStageFlags.PipelineStageTopOfPipeBit, VkPipelineStageFlags.PipelineStageTopOfPipeBit,
                VkDependencyFlags.None,
                null, null, new ImageMemoryBarrier[] { barrier });

            EndSingleTimeCommand(commandBuffer);
        }

        void CopyImage(Image srcImage, Image dstImage, uint width, uint height) {
            var commandBuffer = BeginSingleTimeCommands();

            var subresource = new VkImageSubresourceLayers();
            subresource.aspectMask = VkImageAspectFlags.ImageAspectColorBit;
            subresource.baseArrayLayer = 0;
            subresource.mipLevel = 0;
            subresource.layerCount = 1;

            var region = new VkImageCopy();
            region.srcSubresource = subresource;
            region.dstSubresource = subresource;
            region.srcOffset = new VkOffset3D();
            region.dstOffset = new VkOffset3D();
            region.extent.width = width;
            region.extent.height = height;
            region.extent.depth = 1;

            commandBuffer.Copy(srcImage, VkImageLayout.ImageLayoutTransferSrcOptimal,
                dstImage, VkImageLayout.ImageLayoutTransferDstOptimal,
                new VkImageCopy[] { region });

            EndSingleTimeCommand(commandBuffer);
        }

        void CreateVertexBuffer() {
            ulong bufferSize = (ulong)Interop.SizeOf(vertices);
            Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;
            CreateBuffer(bufferSize,
                VkBufferUsageFlags.BufferUsageTransferSrcBit,
                VkMemoryPropertyFlags.MemoryPropertyHostVisibleBit
                | VkMemoryPropertyFlags.MemoryPropertyHostCoherentBit,
                out stagingBuffer,
                out stagingBufferMemory);

            var data = stagingBufferMemory.Map(0, bufferSize, VkMemoryMapFlags.None);
            Interop.Copy(vertices, data);
            stagingBufferMemory.Unmap();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.BufferUsageTransferDstBit
                | VkBufferUsageFlags.BufferUsageVertexBufferBit,
                VkMemoryPropertyFlags.MemoryPropertyDeviceLocalBit,
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
                VkBufferUsageFlags.BufferUsageTransferSrcBit,
                VkMemoryPropertyFlags.MemoryPropertyHostVisibleBit
                | VkMemoryPropertyFlags.MemoryPropertyHostCoherentBit,
                out stagingBuffer,
                out stagingBufferMemory);

            var data = stagingBufferMemory.Map(0, bufferSize, VkMemoryMapFlags.None);
            Interop.Copy(indices, data);
            stagingBufferMemory.Unmap();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.BufferUsageTransferDstBit
                | VkBufferUsageFlags.BufferUsageIndexBufferBit,
                VkMemoryPropertyFlags.MemoryPropertyDeviceLocalBit,
                out indexBuffer,
                out indexBufferMemory);

            CopyBuffer(stagingBuffer, indexBuffer, bufferSize);

            stagingBuffer.Dispose();
            stagingBufferMemory.Dispose();
        }

        void CreateUniformBuffer() {
            ulong bufferSize = (ulong)Interop.SizeOf<UniformBufferObject>();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.BufferUsageTransferSrcBit,
                VkMemoryPropertyFlags.MemoryPropertyHostVisibleBit
                | VkMemoryPropertyFlags.MemoryPropertyHostCoherentBit,
                out uniformStagingBuffer,
                out uniformStagingBufferMemory);

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.BufferUsageTransferDstBit
                | VkBufferUsageFlags.BufferUsageUniformBufferBit,
                VkMemoryPropertyFlags.MemoryPropertyDeviceLocalBit,
                out uniformBuffer,
                out uniformBufferMemory);
        }

        void CopyBuffer(Buffer src, Buffer dst, ulong size) {
            var buffer = BeginSingleTimeCommands();

            VkBufferCopy region = new VkBufferCopy();
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;

            buffer.Copy(src, dst, new VkBufferCopy[] { region });

            EndSingleTimeCommand(buffer);
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
            var poolSizes = new VkDescriptorPoolSize[2];
            poolSizes[0].type = VkDescriptorType.DescriptorTypeUniformBuffer;
            poolSizes[0].descriptorCount = 1;
            poolSizes[1].type = VkDescriptorType.DescriptorTypeCombinedImageSampler;
            poolSizes[1].descriptorCount = 1;

            var info = new DescriptorPoolCreateInfo();
            info.poolSizes = poolSizes;
            info.maxSets = 1;

            descriptorPool = new DescriptorPool(device, info);
        }

        void CreateDescriptorSet() {
            var layouts = new DescriptorSetLayout[] { descriptorSetLayout };
            var info = new DescriptorSetAllocateInfo();
            info.descriptorSetCount = 1;
            info.setLayouts = layouts;

            descriptorSet = descriptorPool.Allocate(info)[0];

            var bufferInfo = new DescriptorBufferInfo();
            bufferInfo.buffer = uniformBuffer;
            bufferInfo.offset = 0;
            bufferInfo.range = (ulong)Interop.SizeOf<UniformBufferObject>();

            var imageInfo = new DescriptorImageInfo();
            imageInfo.imageLayout = VkImageLayout.ImageLayoutShaderReadOnlyOptimal;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            var descriptorWrites = new WriteDescriptorSet[2];
            descriptorWrites[0] = new WriteDescriptorSet();
            descriptorWrites[0].dstSet = descriptorSet;
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VkDescriptorType.DescriptorTypeUniformBuffer;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].bufferInfo = bufferInfo;

            descriptorWrites[1] = new WriteDescriptorSet();
            descriptorWrites[1].dstSet = descriptorSet;
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VkDescriptorType.DescriptorTypeCombinedImageSampler;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].imageInfo = imageInfo;

            descriptorPool.Update(descriptorWrites);
        }

        void CreateCommandBuffers() {
            if (commandBuffers != null) {
                commandPool.Free(commandBuffers);
            }

            var info = new CommandBufferAllocateInfo();
            info.level = VkCommandBufferLevel.CommandBufferLevelPrimary;
            info.commandBufferCount = (uint)swapchainFramebuffers.Count;

            commandBuffers = new List<CommandBuffer>(commandPool.Allocate(info));

            for (int i = 0; i < commandBuffers.Count; i++) {
                var buffer = commandBuffers[i];
                var beginInfo = new CommandBufferBeginInfo();
                beginInfo.flags = VkCommandBufferUsageFlags.CommandBufferUsageSimultaneousUseBit;

                buffer.Begin(beginInfo);

                var renderPassInfo = new RenderPassBeginInfo();
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapchainFramebuffers[i];
                renderPassInfo.renderArea.extent = swapchainExtent;

                var clearValues = new VkClearValue[2];
                clearValues[0].color.float32_0 = 0;
                clearValues[0].color.float32_1 = 0;
                clearValues[0].color.float32_2 = 0;
                clearValues[0].color.float32_3 = 1f;
                clearValues[1].depthStencil.depth = 1;
                clearValues[1].depthStencil.stencil = 0;

                renderPassInfo.clearValues = clearValues;

                buffer.BeginRenderPass(renderPassInfo, VkSubpassContents.SubpassContentsInline);
                buffer.BindPipeline(VkPipelineBindPoint.PipelineBindPointGraphics, pipeline);
                buffer.BindVertexBuffers(0, new Buffer[] { vertexBuffer }, new ulong[] { 0 });
                buffer.BindIndexBuffer(indexBuffer, 0, VkIndexType.IndexTypeUint32);
                buffer.BindDescriptorSets(VkPipelineBindPoint.PipelineBindPointGraphics, pipelineLayout, 0, new DescriptorSet[] { descriptorSet });
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
