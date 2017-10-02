using System;
using System.IO;
using System.Collections.Generic;
using System.Numerics;

using CSGL;
using CSGL.Graphics;
using CSGL.GLFW;
using CSGL.Vulkan;

namespace AllColors {
    public struct Vertex {
        public Vector3 position;
        public Vector2 uv;

        public Vertex(Vector3 position, Vector2 uv) {
            this.position = position;
            this.uv = uv;
        }

        public static VkVertexInputBindingDescription GetBindingDescription() {
            var result = new VkVertexInputBindingDescription();
            result.binding = 0;
            result.stride = (int)Interop.SizeOf<Vertex>();
            result.inputRate = VkVertexInputRate.Vertex;

            return result;
        }

        public static List<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
            Vertex v = new Vertex();

            var desc1 = new VkVertexInputAttributeDescription();
            desc1.binding = 0;
            desc1.location = 0;
            desc1.format = VkFormat.R32G32B32_Sfloat;
            desc1.offset = (int)Interop.Offset(ref v, ref v.position);

            var desc2 = new VkVertexInputAttributeDescription();
            desc2.binding = 0;
            desc2.location = 1;
            desc2.format = VkFormat.R32G32_Sfloat;
            desc2.offset = (int)Interop.Offset(ref v, ref v.uv);

            return new List<VkVertexInputAttributeDescription> { desc1, desc2 };
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
            using (Program program = new Program()) {
                program.Run();
            }
        }

        const int imageWidth = 2048;
        const int imageHeight = 1024;

        List<string> layers = new List<string> {
            "VK_LAYER_LUNARG_standard_validation",
            //"VK_LAYER_LUNARG_api_dump"
        };

        List<string> deviceExtensions = new List<string> {
            "VK_KHR_swapchain"
        };

        Vertex[] vertices = {
            new Vertex(new Vector3(-imageWidth / 2, -imageHeight / 2, 0), new Vector2(0, 0)),
            new Vertex(new Vector3(imageWidth / 2, -imageHeight / 2, 0),  new Vector2(1, 0)),
            new Vertex(new Vector3(imageWidth / 2, imageHeight / 2, 0),   new Vector2(1, 1)),
            new Vertex(new Vector3(-imageWidth / 2, imageHeight / 2, 0),  new Vector2(0, 1)),
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0
        };

        Window window;
        bool recreateSwapchainFlag;

        ColorSource colorSource;
        Generator generator;

        int graphicsIndex;
        int presentIndex;
        VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;
        IntPtr stagingBufferPtr;
        IntPtr uniformBufferPtr;

        VkInstance instance;
        VkPhysicalDevice physicalDevice;
        VkSurface surface;
        VkDevice device;
        VkSwapchain swapchain;
        List<VkImage> swapchainImages;
        List<VkImageView> swapchainImageViews;
        VkRenderPass renderPass;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipelineLayout pipelineLayout;
        VkGraphicsPipeline pipeline;
        List<VkFramebuffer> swapchainFramebuffers;
        VkCommandPool commandPool;
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VkImage textureImage;
        VkDeviceMemory textureImageMemory;
        VkImageView textureImageView;
        VkSampler textureSampler;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
        VkBuffer uniformBuffer;
        VkDeviceMemory uniformBufferMemory;
        VkDescriptorPool descriptorPool;
        VkDescriptorSet descriptorSet;
        IList<VkCommandBuffer> commandBuffers;
        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;

        void Run() {
            colorSource = new ColorSource(imageWidth, imageHeight, 0);
            generator = new Generator(imageWidth, imageHeight, colorSource);
            CreateWindow();
            CreateInstance();
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
            CreateStagingBuffer();
            CreateTextureImage();
            CreateTextureImageView();
            CreateTextureSampler();
            CreateVertexBuffer();
            CreateIndexBuffer();
            CreateUniformBuffer();
            CreateDescriptorPool();
            CreateDescriptorSet();
            CreateCommandBuffers();
            CreateSyncObjects();
            MainLoop();
        }

        void MainLoop() {
            var waitSemaphores = new List<VkSemaphore> { imageAvailableSemaphore };
            var waitStages = new List<VkPipelineStageFlags> { VkPipelineStageFlags.ColorAttachmentOutputBit };
            var signalSemaphores = new List<VkSemaphore> { renderFinishedSemaphore };
            var swapchains = new List<VkSwapchain> { swapchain };

            var commandBuffers = new List<VkCommandBuffer> { null };
            var index = new List<int> { 0 };

            var submitInfo = new VkSubmitInfo();
            submitInfo.waitSemaphores = waitSemaphores;
            submitInfo.waitDstStageMask = waitStages;
            submitInfo.commandBuffers = commandBuffers;
            submitInfo.signalSemaphores = signalSemaphores;

            var presentInfo = new VkPresentInfo();
            presentInfo.waitSemaphores = signalSemaphores;
            presentInfo.swapchains = swapchains;
            presentInfo.imageIndices = index;

            var submitInfos = new List<VkSubmitInfo> { submitInfo };

            GLFW.ShowWindow(window.Native);
            generator.Start();

            while (!window.ShouldClose) {
                GLFW.PollEvents();

                UpdateUniformBuffer();

                if (recreateSwapchainFlag) {
                    recreateSwapchainFlag = false;
                    RecreateSwapchain();
                }

                int imageIndex;
                var result = swapchain.AcquireNextImage(-1, imageAvailableSemaphore, null, out imageIndex);

                Interop.Copy(generator.Pixels, stagingBufferPtr);
                commandBuffers[0] = this.commandBuffers[(int)imageIndex];

                swapchains[0] = swapchain;
                index[0] = imageIndex;

                graphicsQueue.Submit(submitInfos, null);
                result = presentQueue.Present(presentInfo);
            }

            device.WaitIdle();
        }

        void UpdateUniformBuffer() {
            var ubo = new UniformBufferObject();
            ubo.model = Matrix4x4.Identity;
            ubo.view = Matrix4x4.CreateLookAt(new Vector3(0, 0, -1), new Vector3(), new Vector3(0, 1, 0));
            ubo.proj = Matrix4x4.CreateOrthographic(window.FramebufferWidth, window.FramebufferHeight, -1, 1);

            ulong size = (ulong)Interop.SizeOf<UniformBufferObject>();

            Interop.Write(ubo, uniformBufferPtr);
        }

        void RecreateSwapchain() {
            device.WaitIdle();
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateFramebuffers();
            RecordCommands();
        }

        public void Dispose() {
            generator.Dispose();
            imageAvailableSemaphore.Dispose();
            renderFinishedSemaphore.Dispose();
            descriptorPool.Dispose();
            uniformBuffer.Dispose();
            uniformBufferMemory.Dispose();
            indexBuffer.Dispose();
            indexBufferMemory.Dispose();
            vertexBuffer.Dispose();
            vertexBufferMemory.Dispose();
            textureSampler.Dispose();
            textureImageView.Dispose();
            textureImage.Dispose();
            textureImageMemory.Dispose();
            stagingBuffer.Dispose();
            stagingBufferMemory.Dispose();
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
            window.Dispose();
            GLFW.Terminate();
        }

        void OnWindowResized(int width, int height) {
            if (width == 0 || height == 0) return;
            recreateSwapchainFlag = true;
        }

        void CreateWindow() {
            GLFW.Init();
            GLFW.WindowHint(WindowHint.ClientAPI, (int)ClientAPI.NoAPI);
            GLFW.WindowHint(WindowHint.Maximized, 1);
            GLFW.WindowHint(WindowHint.Visible, 0);
            window = new Window(800, 600, "All Colors");
            window.OnSizeChanged += OnWindowResized;
        }

        void CreateInstance() {
            var extensions = new List<string>(GLFW.GetRequiredInstanceExceptions());

            var appInfo = new VkApplicationInfo {
                apiVersion = new VkVersion(1, 0, 0),
                applicationVersion = new VkVersion(1, 0, 0),
                engineVersion = new VkVersion(1, 0, 0),
                applicationName = "All Colors",
            };

            var info = new VkInstanceCreateInfo {
                applicationInfo = appInfo,
                extensions = extensions,
                layers = layers
            };
            instance = new VkInstance(info);
        }

        void PickPhysicalDevice() {
            physicalDevice = instance.PhysicalDevices[0];
        }

        void CreateSurface() {
            surface = new VkSurface(instance, window);
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

            graphicsIndex = g;
            presentIndex = p;
        }

        void CreateDevice() {
            var features = physicalDevice.Features;

            HashSet<int> uniqueIndices = new HashSet<int> { graphicsIndex, presentIndex };
            List<float> priorities = new List<float> { 1f };
            List<VkDeviceQueueCreateInfo> queueInfos = new List<VkDeviceQueueCreateInfo>(uniqueIndices.Count);

            int i = 0;
            foreach (var ind in uniqueIndices) {
                var queueInfo = new VkDeviceQueueCreateInfo {
                    queueFamilyIndex = ind,
                    queueCount = 1,
                    priorities = priorities
                };

                queueInfos.Add(queueInfo);
                i++;
            }

            var info = new VkDeviceCreateInfo {
                extensions = deviceExtensions,
                queueCreateInfos = queueInfos,
                features = features
            };
            device = new VkDevice(physicalDevice, info);

            graphicsQueue = device.GetQueue(graphicsIndex, 0);
            presentQueue = device.GetQueue(presentIndex, 0);
        }

        SwapchainSupport GetSwapchainSupport(VkPhysicalDevice physicalDevice) {
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
            if (cap.currentExtent.width != -1) {
                return cap.currentExtent;
            } else {
                var extent = new VkExtent2D();
                extent.width = window.FramebufferWidth;
                extent.height = window.FramebufferHeight;

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

            int imageCount = cap.minImageCount + 1;
            if (cap.maxImageCount > 0 && imageCount > cap.maxImageCount) {
                imageCount = cap.maxImageCount;
            }

            var oldSwapchain = swapchain;
            var info = new VkSwapchainCreateInfo();
            info.surface = surface;
            info.oldSwapchain = oldSwapchain;
            info.minImageCount = imageCount;
            info.imageFormat = surfaceFormat.format;
            info.imageColorSpace = surfaceFormat.colorSpace;
            info.imageExtent = extent;
            info.imageArrayLayers = 1;
            info.imageUsage = VkImageUsageFlags.ColorAttachmentBit;

            var queueFamilyIndices = new List<int> { graphicsIndex, presentIndex };

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

            swapchain = new VkSwapchain(device, info);
            oldSwapchain?.Dispose();

            swapchainImages = new List<VkImage>(swapchain.Images);

            swapchainImageFormat = surfaceFormat.format;
            swapchainExtent = extent;
        }

        void CreateImageView(VkImage image, VkFormat format, ref VkImageView imageView) {
            var info = new VkImageViewCreateInfo();
            info.image = image;
            info.viewType = VkImageViewType._2D;
            info.format = format;
            info.subresourceRange.aspectMask = VkImageAspectFlags.ColorBit;
            info.subresourceRange.baseMipLevel = 0; ;
            info.subresourceRange.levelCount = 1;
            info.subresourceRange.baseArrayLayer = 0;
            info.subresourceRange.layerCount = 1;

            imageView?.Dispose();
            imageView = new VkImageView(device, info);
        }

        void CreateImageViews() {
            if (swapchainImageViews != null) {
                foreach (var iv in swapchainImageViews) iv.Dispose();
            }

            swapchainImageViews = new List<VkImageView>();
            foreach (var image in swapchainImages) {
                VkImageView temp = null;
                CreateImageView(image, swapchainImageFormat, ref temp);
                swapchainImageViews.Add(temp);
            }
        }

        void CreateRenderPass() {
            var colorAttachment = new VkAttachmentDescription();
            colorAttachment.format = swapchainImageFormat;
            colorAttachment.samples = VkSampleCountFlags._1_Bit;
            colorAttachment.loadOp = VkAttachmentLoadOp.Clear;
            colorAttachment.storeOp = VkAttachmentStoreOp.Store;
            colorAttachment.stencilLoadOp = VkAttachmentLoadOp.DontCare;
            colorAttachment.stencilStoreOp = VkAttachmentStoreOp.DontCare;
            colorAttachment.initialLayout = VkImageLayout.Undefined;
            colorAttachment.finalLayout = VkImageLayout.PresentSrcKhr;

            var colorAttachmentRef = new VkAttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VkImageLayout.ColorAttachmentOptimal;

            var subpass = new VkSubpassDescription();
            subpass.pipelineBindPoint = VkPipelineBindPoint.Graphics;
            subpass.colorAttachments = new List<VkAttachmentReference> { colorAttachmentRef };

            var dependency = new VkSubpassDependency();
            dependency.srcSubpass = -1;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VkPipelineStageFlags.BottomOfPipeBit;
            dependency.srcAccessMask = VkAccessFlags.MemoryReadBit;
            dependency.dstStageMask = VkPipelineStageFlags.ColorAttachmentOutputBit;
            dependency.dstAccessMask = VkAccessFlags.ColorAttachmentReadBit
                                    | VkAccessFlags.ColorAttachmentWriteBit;

            var info = new VkRenderPassCreateInfo();
            info.attachments = new List<VkAttachmentDescription> { colorAttachment };
            info.subpasses = new List<VkSubpassDescription> { subpass };
            info.dependencies = new List<VkSubpassDependency> { dependency };

            renderPass?.Dispose();
            renderPass = new VkRenderPass(device, info);
        }

        void CreateDescriptorSetLayout() {
            var uboLayoutBinding = new VkDescriptorSetLayoutBinding();
            uboLayoutBinding.binding = 0;
            uboLayoutBinding.descriptorType = VkDescriptorType.UniformBuffer;
            uboLayoutBinding.descriptorCount = 1;
            uboLayoutBinding.stageFlags = VkShaderStageFlags.VertexBit;

            var samplerLayoutBinding = new VkDescriptorSetLayoutBinding();
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VkDescriptorType.CombinedImageSampler;
            samplerLayoutBinding.stageFlags = VkShaderStageFlags.FragmentBit;

            var info = new VkDescriptorSetLayoutCreateInfo();
            info.bindings = new List<VkDescriptorSetLayoutBinding> { uboLayoutBinding, samplerLayoutBinding };

            descriptorSetLayout = new VkDescriptorSetLayout(device, info);
        }

        public VkShaderModule CreateShaderModule(byte[] code) {
            var info = new VkShaderModuleCreateInfo();
            info.data = code;
            return new VkShaderModule(device, info);
        }

        void CreateGraphicsPipeline() {
            var vert = CreateShaderModule(File.ReadAllBytes("vert.spv"));
            var frag = CreateShaderModule(File.ReadAllBytes("frag.spv"));

            var vertInfo = new VkPipelineShaderStageCreateInfo();
            vertInfo.stage = VkShaderStageFlags.VertexBit;
            vertInfo.module = vert;
            vertInfo.name = "main";

            var fragInfo = new VkPipelineShaderStageCreateInfo();
            fragInfo.stage = VkShaderStageFlags.FragmentBit;
            fragInfo.module = frag;
            fragInfo.name = "main";

            var shaderStages = new List<VkPipelineShaderStageCreateInfo> { vertInfo, fragInfo };

            var vertexInputInfo = new VkPipelineVertexInputStateCreateInfo();
            vertexInputInfo.vertexBindingDescriptions = new List<VkVertexInputBindingDescription> { Vertex.GetBindingDescription() };
            vertexInputInfo.vertexAttributeDescriptions = Vertex.GetAttributeDescriptions();

            var inputAssembly = new VkPipelineInputAssemblyStateCreateInfo();
            inputAssembly.topology = VkPrimitiveTopology.TriangleList;

            var viewport = new VkViewport();
            viewport.width = swapchainExtent.width;
            viewport.height = swapchainExtent.height;
            viewport.minDepth = 0f;
            viewport.maxDepth = 1f;

            var scissor = new VkRect2D();
            scissor.extent = swapchainExtent;

            var viewportState = new VkPipelineViewportStateCreateInfo();
            viewportState.viewports = new List<VkViewport> { viewport };
            viewportState.scissors = new List<VkRect2D> { scissor };

            var rasterizer = new VkPipelineRasterizationStateCreateInfo();
            rasterizer.polygonMode = VkPolygonMode.Fill;
            rasterizer.lineWidth = 1f;
            rasterizer.cullMode = VkCullModeFlags.BackBit;
            rasterizer.frontFace = VkFrontFace.CounterClockwise;

            var multisampling = new VkPipelineMultisampleStateCreateInfo();
            multisampling.rasterizationSamples = VkSampleCountFlags._1_Bit;
            multisampling.minSampleShading = 1f;

            var colorBlendAttachment = new VkPipelineColorBlendAttachmentState();
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

            var colorBlending = new VkPipelineColorBlendStateCreateInfo();
            colorBlending.logicOp = VkLogicOp.Copy;
            colorBlending.attachments = new List<VkPipelineColorBlendAttachmentState> { colorBlendAttachment };

            var dynamic = new VkPipelineDynamicStateCreateInfo();
            dynamic.dynamicStates = new List<VkDynamicState> {
                VkDynamicState.Viewport,
                VkDynamicState.Scissor
            };

            var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo();
            pipelineLayoutInfo.setLayouts = new List<VkDescriptorSetLayout> { descriptorSetLayout };

            pipelineLayout?.Dispose();

            pipelineLayout = new VkPipelineLayout(device, pipelineLayoutInfo);

            var info = new VkGraphicsPipelineCreateInfo();
            info.stages = shaderStages;
            info.vertexInputState = vertexInputInfo;
            info.inputAssemblyState = inputAssembly;
            info.viewportState = viewportState;
            info.rasterizationState = rasterizer;
            info.multisampleState = multisampling;
            info.colorBlendState = colorBlending;
            info.dynamicState = dynamic;
            info.layout = pipelineLayout;
            info.renderPass = renderPass;
            info.subpass = 0;
            info.basePipelineHandle = null;
            info.basePipelineIndex = -1;

            pipeline?.Dispose();

            pipeline = new VkGraphicsPipeline(device, info, null);

            vert.Dispose();
            frag.Dispose();
        }

        void CreateFramebuffers() {
            if (swapchainFramebuffers != null) {
                foreach (var fb in swapchainFramebuffers) fb.Dispose();
            }

            swapchainFramebuffers = new List<VkFramebuffer>(swapchainImageViews.Count);

            for (int i = 0; i < swapchainImageViews.Count; i++) {
                var attachments = new List<VkImageView> { swapchainImageViews[i] };

                var info = new VkFramebufferCreateInfo();
                info.renderPass = renderPass;
                info.attachments = attachments;
                info.width = swapchainExtent.width;
                info.height = swapchainExtent.height;
                info.layers = 1;

                swapchainFramebuffers.Add(new VkFramebuffer(device, info));
            }
        }

        void CreateCommandPool() {
            var info = new VkCommandPoolCreateInfo();
            info.queueFamilyIndex = graphicsIndex;
            info.flags = VkCommandPoolCreateFlags.ResetCommandBufferBit;

            commandPool = new VkCommandPool(device, info);
        }

        int FindMemoryType(uint filter, VkMemoryPropertyFlags flags) {
            var props = physicalDevice.MemoryProperties;

            for (int i = 0; i < props.MemoryTypes.Count; i++) {
                if ((filter & (1 << i)) != 0 && (props.MemoryTypes[i].propertyFlags & flags) == flags) {
                    return i;
                }
            }

            throw new Exception("Failed to find suitable memory type");
        }

        void CreateBuffer(long size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, out VkBuffer buffer, out VkDeviceMemory memory) {
            var info = new VkBufferCreateInfo();
            info.size = size;
            info.usage = usage;
            info.sharingMode = VkSharingMode.Exclusive;

            buffer = new VkBuffer(device, info);

            var allocInfo = new VkMemoryAllocateInfo();
            allocInfo.allocationSize = buffer.Requirements.size;
            allocInfo.memoryTypeIndex = FindMemoryType(buffer.Requirements.memoryTypeBits, properties);

            memory = new VkDeviceMemory(device, allocInfo);
            buffer.Bind(memory, 0);
        }

        void CreateStagingBuffer() {
            long imageSize = imageWidth * imageHeight * 4;
            CreateBuffer(imageSize, VkBufferUsageFlags.TransferSrcBit,
                VkMemoryPropertyFlags.HostVisibleBit | VkMemoryPropertyFlags.HostCoherentBit,
                out stagingBuffer, out stagingBufferMemory);

            stagingBufferPtr = stagingBufferMemory.Map(0, imageSize);   //DeviceMemory is implicitly unmapped when it is freed
        }

        void CreateImage(int width, int height,
            VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
            out VkImage image, out VkDeviceMemory memory) {

            var info = new VkImageCreateInfo();
            info.imageType = VkImageType._2D;
            info.extent.width = width;
            info.extent.height = height;
            info.extent.depth = 1;
            info.mipLevels = 1;
            info.arrayLayers = 1;
            info.format = format;
            info.tiling = tiling;
            info.initialLayout = VkImageLayout.Undefined;
            info.usage = usage;
            info.sharingMode = VkSharingMode.Exclusive;
            info.samples = VkSampleCountFlags._1_Bit;

            image = new VkImage(device, info);

            var req = image.Requirements;

            var allocInfo = new VkMemoryAllocateInfo();
            allocInfo.allocationSize = req.size;
            allocInfo.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, properties);

            memory = new VkDeviceMemory(device, allocInfo);

            image.Bind(memory, 0);
        }

        VkCommandBuffer BeginSingleTimeCommands() {
            var commandBuffer = commandPool.Allocate(VkCommandBufferLevel.Primary);

            var beginInfo = new VkCommandBufferBeginInfo();
            beginInfo.flags = VkCommandBufferUsageFlags.OneTimeSubmitBit;

            commandBuffer.Begin(beginInfo);

            return commandBuffer;
        }

        void EndSingleTimeCommand(VkCommandBuffer commandBuffer) {
            commandBuffer.End();
            var commands = new List<VkCommandBuffer> { commandBuffer };

            var info = new VkSubmitInfo();
            info.commandBuffers = commands;

            graphicsQueue.Submit(new List<VkSubmitInfo> { info }, null);
            graphicsQueue.WaitIdle();

            commandPool.Free(commands);
        }

        void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
            var commandBuffer = BeginSingleTimeCommands();

            var barrier = new VkImageMemoryBarrier();
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.srcQueueFamilyIndex = -1;    //VK_QUEUE_FAMILY_IGNORED
            barrier.dstQueueFamilyIndex = -1;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = VkImageAspectFlags.ColorBit;
            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags source;
            VkPipelineStageFlags dest;

            if (oldLayout == VkImageLayout.Undefined && newLayout == VkImageLayout.ShaderReadOnlyOptimal) {
                barrier.srcAccessMask = VkAccessFlags.None;
                barrier.dstAccessMask = VkAccessFlags.ShaderReadBit;

                source = VkPipelineStageFlags.TopOfPipeBit;
                dest = VkPipelineStageFlags.FragmentShaderBit;
            } else {
                throw new Exception("Unsupported transition");
            }

            commandBuffer.PipelineBarrier(
                source, dest,
                VkDependencyFlags.None,
                null, null, new List<VkImageMemoryBarrier> { barrier });

            EndSingleTimeCommand(commandBuffer);
        }

        void CreateTextureImage() {
            CreateImage(imageWidth, imageHeight,
                VkFormat.R8G8B8A8_Unorm,
                VkImageTiling.Optimal,
                VkImageUsageFlags.TransferDstBit | VkImageUsageFlags.SampledBit,
                VkMemoryPropertyFlags.DeviceLocalBit,
                out textureImage, out textureImageMemory);

            TransitionImageLayout(textureImage, VkFormat.R8G8B8A8_Unorm,
                VkImageLayout.Undefined, VkImageLayout.ShaderReadOnlyOptimal);
        }

        void CreateTextureImageView() {
            CreateImageView(textureImage, VkFormat.R8G8B8A8_Unorm, ref textureImageView);
        }

        void CreateTextureSampler() {
            var info = new VkSamplerCreateInfo();
            info.magFilter = VkFilter.Nearest;
            info.minFilter = VkFilter.Nearest;
            info.addressModeU = VkSamplerAddressMode.Repeat;
            info.addressModeV = VkSamplerAddressMode.Repeat;
            info.addressModeW = VkSamplerAddressMode.Repeat;
            info.anisotropyEnable = true;
            info.maxAnisotropy = 16;
            info.borderColor = VkBorderColor.FloatOpaqueBlack;
            info.unnormalizedCoordinates = false;

            textureSampler = new VkSampler(device, info);
        }

        void CopyBuffer(VkBuffer src, VkBuffer dst, long size) {
            var buffer = BeginSingleTimeCommands();

            VkBufferCopy region = new VkBufferCopy();
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;

            buffer.CopyBuffer(src, dst, new VkBufferCopy[] { region });

            EndSingleTimeCommand(buffer);
        }

        void CreateVertexBuffer() {
            long bufferSize = Interop.SizeOf(vertices);
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
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
            long bufferSize = Interop.SizeOf(indices);
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingBufferMemory;
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
            long bufferSize = Interop.SizeOf<UniformBufferObject>();

            CreateBuffer(bufferSize,
                VkBufferUsageFlags.UniformBufferBit,
                VkMemoryPropertyFlags.HostVisibleBit
                | VkMemoryPropertyFlags.HostCoherentBit,
                out uniformBuffer,
                out uniformBufferMemory);

            uniformBufferPtr = uniformBufferMemory.Map(0, bufferSize);
        }

        void CreateDescriptorPool() {
            var size1 = new VkDescriptorPoolSize();
            size1.type = VkDescriptorType.UniformBuffer;
            size1.descriptorCount = 1;

            var size2 = new VkDescriptorPoolSize();
            size2.type = VkDescriptorType.CombinedImageSampler;
            size2.descriptorCount = 1;

            var poolSizes = new List<VkDescriptorPoolSize> { size1, size2 };

            var info = new VkDescriptorPoolCreateInfo();
            info.poolSizes = poolSizes;
            info.maxSets = 1;

            descriptorPool = new VkDescriptorPool(device, info);
        }

        void CreateDescriptorSet() {
            var layouts = new List<VkDescriptorSetLayout> { descriptorSetLayout };
            var info = new VkDescriptorSetAllocateInfo();
            info.setLayouts = layouts;

            descriptorSet = descriptorPool.Allocate(info)[0];

            var bufferInfo = new VkDescriptorBufferInfo();
            bufferInfo.buffer = uniformBuffer;
            bufferInfo.offset = 0;
            bufferInfo.range = Interop.SizeOf<UniformBufferObject>();

            var imageInfo = new VkDescriptorImageInfo();
            imageInfo.imageLayout = VkImageLayout.ShaderReadOnlyOptimal;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            var descriptorWrites = new List<VkWriteDescriptorSet>();
            descriptorWrites.Add(new VkWriteDescriptorSet());
            descriptorWrites[0].dstSet = descriptorSet;
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VkDescriptorType.UniformBuffer;
            descriptorWrites[0].bufferInfo = new List<VkDescriptorBufferInfo> { bufferInfo };

            descriptorWrites.Add(new VkWriteDescriptorSet());
            descriptorWrites[1].dstSet = descriptorSet;
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VkDescriptorType.CombinedImageSampler;
            descriptorWrites[1].imageInfo = new List<VkDescriptorImageInfo> { imageInfo };

            descriptorSet.Update(descriptorWrites, null);
        }

        void CreateCommandBuffers() {
            VkCommandBufferAllocateInfo info = new VkCommandBufferAllocateInfo {
                commandBufferCount = swapchainImages.Count,
                level = VkCommandBufferLevel.Primary
            };
            commandBuffers = commandPool.Allocate(info);
        }

        void RecordCommands() {
            for (int i = 0; i < swapchainImages.Count; i++) {
                VkCommandBuffer commandBuffer = commandBuffers[i];
                commandBuffer.Reset(VkCommandBufferResetFlags.None);
                VkCommandBufferBeginInfo beginInfo = new VkCommandBufferBeginInfo();
                beginInfo.flags = VkCommandBufferUsageFlags.SimultaneousUseBit;
                commandBuffer.Begin(beginInfo);

                //transfer to writable
                commandBuffer.PipelineBarrier(VkPipelineStageFlags.FragmentShaderBit, VkPipelineStageFlags.TransferBit, VkDependencyFlags.None,
                    null, null,
                    new List<VkImageMemoryBarrier> {
                    new VkImageMemoryBarrier {
                        image = textureImage,
                        oldLayout = VkImageLayout.ShaderReadOnlyOptimal,
                        newLayout = VkImageLayout.TransferDstOptimal,
                        srcQueueFamilyIndex = -1,    //VK_QUEUE_FAMILY_IGNORED
                        dstQueueFamilyIndex = -1,
                        srcAccessMask = VkAccessFlags.None,
                        dstAccessMask = VkAccessFlags.TransferWriteBit,
                        subresourceRange = new VkImageSubresourceRange {
                            aspectMask = VkImageAspectFlags.ColorBit,
                            baseArrayLayer = 0,
                            layerCount = 1,
                            baseMipLevel = 0,
                            levelCount = 1
                        }
                    }
                    });

                commandBuffer.CopyBufferToImage(stagingBuffer, textureImage, VkImageLayout.TransferDstOptimal,
                    new VkBufferImageCopy[] {
                    new VkBufferImageCopy {
                        imageExtent = new VkExtent3D {
                            width = imageWidth,
                            height = imageHeight,
                            depth = 1
                        },
                        imageSubresource = new VkImageSubresourceLayers {
                            aspectMask = VkImageAspectFlags.ColorBit,
                            baseArrayLayer = 0,
                            layerCount = 1,
                            mipLevel = 0
                        }
                    }
                    });

                //transfer to shader readable
                commandBuffer.PipelineBarrier(VkPipelineStageFlags.TransferBit, VkPipelineStageFlags.FragmentShaderBit, VkDependencyFlags.None,
                    null, null,
                    new List<VkImageMemoryBarrier> {
                    new VkImageMemoryBarrier {
                        image = textureImage,
                        oldLayout = VkImageLayout.TransferDstOptimal,
                        newLayout = VkImageLayout.ShaderReadOnlyOptimal,
                        srcQueueFamilyIndex = -1,    //VK_QUEUE_FAMILY_IGNORED
                        dstQueueFamilyIndex = -1,
                        srcAccessMask = VkAccessFlags.TransferWriteBit,
                        dstAccessMask = VkAccessFlags.ShaderReadBit,
                        subresourceRange = new VkImageSubresourceRange {
                            aspectMask = VkImageAspectFlags.ColorBit,
                            baseArrayLayer = 0,
                            layerCount = 1,
                            baseMipLevel = 0,
                            levelCount = 1
                        }
                    }
                    });

                VkRenderPassBeginInfo renderPassBeginInfo = new VkRenderPassBeginInfo();
                renderPassBeginInfo.renderPass = renderPass;
                renderPassBeginInfo.framebuffer = swapchainFramebuffers[i];
                renderPassBeginInfo.renderArea = new VkRect2D {
                    extent = new VkExtent2D {
                        width = window.FramebufferWidth,
                        height = window.FramebufferHeight
                    }
                };
                renderPassBeginInfo.clearValues = new List<VkClearValue> {
                new VkClearValue {
                    color = new VkClearColorValue(0.125f, 0.125f, 0.125f, 0.125f)
                }
            };

                commandBuffer.BeginRenderPass(renderPassBeginInfo, VkSubpassContents.Inline);

                commandBuffer.BindPipeline(VkPipelineBindPoint.Graphics, pipeline);
                commandBuffer.BindDescriptorSets(VkPipelineBindPoint.Graphics, pipelineLayout, 0, descriptorSet, null);
                commandBuffer.BindVertexBuffers(0, vertexBuffer, 0);
                commandBuffer.BindIndexBuffer(indexBuffer, 0, VkIndexType.UINT32);

                commandBuffer.SetViewports(0, new VkViewport {
                    width = swapchainExtent.width,
                    height = swapchainExtent.height,
                    minDepth = 0,
                    maxDepth = 1,
                });
                commandBuffer.SetScissor(0, new VkRect2D {
                    extent = swapchainExtent
                });
                commandBuffer.DrawIndexed(6, 1, 0, 0, 0);

                commandBuffer.EndRenderPass();
                commandBuffer.End();
            }
        }

        void CreateSyncObjects() {
            imageAvailableSemaphore = new VkSemaphore(device);
            renderFinishedSemaphore = new VkSemaphore(device);
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
