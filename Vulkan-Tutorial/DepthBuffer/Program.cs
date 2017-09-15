using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using System.Diagnostics;

using CSGL;
using CSGL.STB;
using CSGL.GLFW;
using CSGL.Vulkan;
using CSGL.GLFW.Unmanaged;

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
            result.inputRate = VkVertexInputRate.Vertex;

            return result;
        }

        public static List<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
            Vertex v = new Vertex();

            var desc1 = new VkVertexInputAttributeDescription();
            desc1.binding = 0;
            desc1.location = 0;
            desc1.format = VkFormat.R32G32B32_Sfloat;
            desc1.offset = (uint)Interop.Offset(ref v, ref v.position);

            var desc2 = new VkVertexInputAttributeDescription();
            desc2.binding = 0;
            desc2.location = 1;
            desc2.format = VkFormat.R32G32B32_Sfloat;
            desc2.offset = (uint)Interop.Offset(ref v, ref v.color);

            var desc3 = new VkVertexInputAttributeDescription();
            desc3.binding = 0;
            desc3.location = 2;
            desc3.format = VkFormat.R32G32_Sfloat;
            desc3.offset = (uint)Interop.Offset(ref v, ref v.texCoord);

            return new List<VkVertexInputAttributeDescription> { desc1, desc2, desc3 };
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
            window = GLFW.CreateWindow(width, height, "Depth Buffer", MonitorPtr.Null, WindowPtr.Null);
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
                applicationName = "Depth Buffer"
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

        void CreateImageView(Image image, VkFormat format, VkImageAspectFlags aspectFlags, ref ImageView imageView) {
            var info = new ImageViewCreateInfo();
            info.image = image;
            info.viewType = VkImageViewType._2D;
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
                CreateImageView(image, swapchainImageFormat, VkImageAspectFlags.ColorBit, ref temp);
                swapchainImageViews.Add(temp);
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

            var depthAttachment = new AttachmentDescription();
            depthAttachment.format = FindDepthFormat();
            depthAttachment.samples = VkSampleCountFlags._1_Bit;
            depthAttachment.loadOp = VkAttachmentLoadOp.Clear;
            depthAttachment.storeOp = VkAttachmentStoreOp.DontCare;
            depthAttachment.stencilLoadOp = VkAttachmentLoadOp.DontCare;
            depthAttachment.stencilStoreOp = VkAttachmentStoreOp.DontCare;
            depthAttachment.initialLayout = VkImageLayout.DepthStencilAttachmentOptimal;
            depthAttachment.finalLayout = VkImageLayout.DepthStencilAttachmentOptimal;

            var colorAttachmentRef = new AttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VkImageLayout.ColorAttachmentOptimal;

            var depthAttachmentRef = new AttachmentReference();
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VkImageLayout.DepthStencilAttachmentOptimal;

            var subpass = new SubpassDescription();
            subpass.pipelineBindPoint = VkPipelineBindPoint.Graphics;
            subpass.colorAttachments = new List<AttachmentReference> { colorAttachmentRef };
            subpass.depthStencilAttachment = depthAttachmentRef;

            var dependency = new SubpassDependency();
            dependency.srcSubpass = uint.MaxValue;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VkPipelineStageFlags.BottomOfPipeBit;
            dependency.srcAccessMask = VkAccessFlags.MemoryReadBit;
            dependency.dstStageMask = VkPipelineStageFlags.ColorAttachmentOutputBit;
            dependency.dstAccessMask = VkAccessFlags.ColorAttachmentReadBit
                                    | VkAccessFlags.ColorAttachmentWriteBit;

            var info = new RenderPassCreateInfo();
            info.attachments = new List<AttachmentDescription> { colorAttachment, depthAttachment };
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

            var samplerLayoutBinding = new VkDescriptorSetLayoutBinding();
            samplerLayoutBinding.binding = 1;
            samplerLayoutBinding.descriptorCount = 1;
            samplerLayoutBinding.descriptorType = VkDescriptorType.CombinedImageSampler;
            samplerLayoutBinding.stageFlags = VkShaderStageFlags.FragmentBit;

            var info = new DescriptorSetLayoutCreateInfo();
            info.bindings = new List<VkDescriptorSetLayoutBinding> { uboLayoutBinding, samplerLayoutBinding };

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

            var depthStencil = new PipelineDepthStencilStateCreateInfo();
            depthStencil.depthTestEnable = true;
            depthStencil.depthWriteEnable = true;
            depthStencil.depthCompareOp = VkCompareOp.Less;
            depthStencil.depthBoundsTestEnable = false;
            depthStencil.minDepthBounds = 0;
            depthStencil.maxDepthBounds = 1;
            depthStencil.stencilTestEnable = false;

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
            info.depthStencilState = depthStencil;
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
                var attachments = new List<ImageView> { swapchainImageViews[i], depthImageView };

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

        void CreateDepthResources() {
            depthImage?.Dispose();
            depthImageMemory?.Dispose();
            depthImageView?.Dispose();

            VkFormat depthFormat = FindDepthFormat();
            CreateImage(swapchainExtent.width, swapchainExtent.height, depthFormat,
                VkImageTiling.Optimal, VkImageUsageFlags.DepthStencilAttachmentBit,
                VkMemoryPropertyFlags.DeviceLocalBit,
                out depthImage, out depthImageMemory);

            CreateImageView(depthImage, depthFormat, VkImageAspectFlags.DepthBit, ref depthImageView);

            TransitionImageLayout(depthImage, depthFormat, VkImageLayout.Undefined, VkImageLayout.DepthStencilAttachmentOptimal);
        }

        VkFormat FindSupportedFormat(List<VkFormat> candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {
            foreach (var format in candidates) {
                var props = physicalDevice.GetFormatProperties(format);

                if (tiling == VkImageTiling.Linear && (props.linearTilingFeatures & features) == features) {
                    return format;
                } else if (tiling == VkImageTiling.Optimal && (props.optimalTilingFeatures & features) == features) {
                    return format;
                }
            }

            throw new Exception("Failed to find supported format");
        }

        VkFormat FindDepthFormat() {
            return FindSupportedFormat(
                new List<VkFormat>() { VkFormat.D32_Sfloat, VkFormat.D32_Sfloat_S8_Uint, VkFormat.D24_Unorm_S8_Uint },
                VkImageTiling.Optimal,
                VkFormatFeatureFlags.DepthStencilAttachmentBit);
        }

        bool HasStencilComponent(VkFormat format) {
            return format == VkFormat.D32_Sfloat_S8_Uint || format == VkFormat.D24_Unorm_S8_Uint;
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

        void CreateTextureImage() {
            byte[] texData = File.ReadAllBytes(image);
            int width;
            int height;
            int comp;
            var pixels = STB.Load(texData, out width, out height, out comp, 4);

            ulong imageSize = (ulong)(width * height * 4);

            Buffer stagingBuffer;
            DeviceMemory stagingBufferMemory;

            CreateBuffer(imageSize, VkBufferUsageFlags.TransferSrcBit,
                VkMemoryPropertyFlags.HostVisibleBit | VkMemoryPropertyFlags.HostCoherentBit,
                out stagingBuffer, out stagingBufferMemory);

            var data = stagingBufferMemory.Map(0, imageSize);
            Interop.Copy(pixels, data, (int)imageSize);
            stagingBufferMemory.Unmap();

            CreateImage((uint)width, (uint)height,
                VkFormat.R8G8B8A8_Unorm,
                VkImageTiling.Optimal,
                VkImageUsageFlags.TransferDstBit | VkImageUsageFlags.SampledBit,
                VkMemoryPropertyFlags.DeviceLocalBit,
                out textureImage, out textureImageMemory);

            TransitionImageLayout(textureImage, VkFormat.R8G8B8A8_Unorm,
                VkImageLayout.Undefined, VkImageLayout.TransferDstOptimal);
            CopyBufferToImage(stagingBuffer, textureImage, (uint)width, (uint)height);

            TransitionImageLayout(textureImage, VkFormat.R8G8B8A8_Unorm,
                VkImageLayout.TransferDstOptimal, VkImageLayout.ShaderReadOnlyOptimal);

            stagingBuffer.Dispose();
            stagingBufferMemory.Dispose();
        }

        void CreateTextureImageView() {
            CreateImageView(textureImage, VkFormat.R8G8B8A8_Unorm, VkImageAspectFlags.ColorBit, ref textureImageView);
        }

        void CreateTextureSampler() {
            var info = new SamplerCreateInfo();
            info.magFilter = VkFilter.Linear;
            info.minFilter = VkFilter.Linear;
            info.addressModeU = VkSamplerAddressMode.Repeat;
            info.addressModeV = VkSamplerAddressMode.Repeat;
            info.addressModeW = VkSamplerAddressMode.Repeat;
            info.anisotropyEnable = true;
            info.maxAnisotropy = 16;
            info.borderColor = VkBorderColor.FloatOpaqueBlack;
            info.unnormalizedCoordinates = false;

            textureSampler = new Sampler(device, info);
        }

        void CreateImage(uint width, uint height,
            VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
            out Image image, out DeviceMemory memory) {

            var info = new ImageCreateInfo();
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

            image = new Image(device, info);

            var req = image.Requirements;

            var allocInfo = new MemoryAllocateInfo();
            allocInfo.allocationSize = req.size;
            allocInfo.memoryTypeIndex = FindMemoryType(req.memoryTypeBits, properties);

            memory = new DeviceMemory(device, allocInfo);

            image.Bind(memory, 0);
        }

        CommandBuffer BeginSingleTimeCommands() {
            var commandBuffer = commandPool.Allocate(VkCommandBufferLevel.Primary);

            var beginInfo = new CommandBufferBeginInfo();
            beginInfo.flags = VkCommandBufferUsageFlags.OneTimeSubmitBit;

            commandBuffer.Begin(beginInfo);

            return commandBuffer;
        }

        void EndSingleTimeCommand(CommandBuffer commandBuffer) {
            commandBuffer.End();
            var commands = new List<CommandBuffer> { commandBuffer };

            var info = new SubmitInfo();
            info.commandBuffers = commands;

            graphicsQueue.Submit(new List<SubmitInfo> { info }, null);
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

            if (newLayout == VkImageLayout.DepthStencilAttachmentOptimal) {
                barrier.subresourceRange.aspectMask = VkImageAspectFlags.DepthBit;

                if (HasStencilComponent(format)) {
                    barrier.subresourceRange.aspectMask |= VkImageAspectFlags.StencilBit;
                }
            } else {
                barrier.subresourceRange.aspectMask = VkImageAspectFlags.ColorBit;
            }


            barrier.subresourceRange.baseMipLevel = 0;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkPipelineStageFlags source;
            VkPipelineStageFlags dest;

            if (oldLayout == VkImageLayout.Undefined && newLayout == VkImageLayout.TransferSrcOptimal) {
                barrier.srcAccessMask = VkAccessFlags.None;
                barrier.dstAccessMask = VkAccessFlags.TransferReadBit;

                source = VkPipelineStageFlags.TopOfPipeBit;
                dest = VkPipelineStageFlags.TransferBit;
            } else if (oldLayout == VkImageLayout.Undefined && newLayout == VkImageLayout.TransferDstOptimal) {
                barrier.srcAccessMask = VkAccessFlags.None;
                barrier.dstAccessMask = VkAccessFlags.TransferWriteBit;

                source = VkPipelineStageFlags.TopOfPipeBit;
                dest = VkPipelineStageFlags.TransferBit;
            } else if (oldLayout == VkImageLayout.TransferDstOptimal && newLayout == VkImageLayout.ShaderReadOnlyOptimal) {
                barrier.srcAccessMask = VkAccessFlags.TransferWriteBit;
                barrier.dstAccessMask = VkAccessFlags.ShaderReadBit;

                source = VkPipelineStageFlags.TransferBit;
                dest = VkPipelineStageFlags.FragmentShaderBit;
            } else if (oldLayout == VkImageLayout.Undefined && newLayout == VkImageLayout.DepthStencilAttachmentOptimal) {
                barrier.srcAccessMask = VkAccessFlags.None;
                barrier.dstAccessMask = VkAccessFlags.DepthStencilAttachmentReadBit | VkAccessFlags.DepthStencilAttachmentWriteBit;

                source = VkPipelineStageFlags.TopOfPipeBit;
                dest = VkPipelineStageFlags.EarlyFragmentTestsBit;
            } else {
                throw new Exception("Unsupported layout transition");
            }

            commandBuffer.PipelineBarrier(
                source, dest,
                VkDependencyFlags.None,
                null, null, new List<ImageMemoryBarrier> { barrier });

            EndSingleTimeCommand(commandBuffer);
        }

        void CopyBufferToImage(Buffer buffer, Image image, uint width, uint height) {
            CommandBuffer commandBuffer = BeginSingleTimeCommands();

            VkBufferImageCopy region = new VkBufferImageCopy();
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VkImageAspectFlags.ColorBit;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;

            region.imageOffset = new VkOffset3D();
            region.imageExtent = new VkExtent3D {
                width = width,
                height = height,
                depth = 1
            };

            commandBuffer.CopyBufferToImage(buffer, image, VkImageLayout.TransferDstOptimal, new VkBufferImageCopy[] { region });

            EndSingleTimeCommand(commandBuffer);
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
            var buffer = BeginSingleTimeCommands();

            VkBufferCopy region = new VkBufferCopy();
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;

            buffer.CopyBuffer(src, dst, new VkBufferCopy[] { region });

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
            var size1 = new VkDescriptorPoolSize();
            size1.type = VkDescriptorType.UniformBuffer;
            size1.descriptorCount = 1;

            var size2 = new VkDescriptorPoolSize();
           size2.type = VkDescriptorType.CombinedImageSampler;
           size2.descriptorCount = 1;

            var poolSizes = new List<VkDescriptorPoolSize> { size1, size2};

            var info = new DescriptorPoolCreateInfo();
            info.poolSizes = poolSizes;
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

            var imageInfo = new DescriptorImageInfo();
            imageInfo.imageLayout = VkImageLayout.ShaderReadOnlyOptimal;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            var descriptorWrites = new List<WriteDescriptorSet>();
            descriptorWrites.Add(new WriteDescriptorSet());
            descriptorWrites[0].dstSet = descriptorSet;
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VkDescriptorType.UniformBuffer;
            descriptorWrites[0].bufferInfo = new List<DescriptorBufferInfo> { bufferInfo };

            descriptorWrites.Add(new WriteDescriptorSet());
            descriptorWrites[1].dstSet = descriptorSet;
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VkDescriptorType.CombinedImageSampler;
            descriptorWrites[1].imageInfo = new List<DescriptorImageInfo> { imageInfo };

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

                var clear1 = new VkClearValue();
                clear1.color.float32_0 = 0;
                clear1.color.float32_1 = 0;
                clear1.color.float32_2 = 0;
                clear1.color.float32_3 = 1f;
                var clear2 = new VkClearValue();
                clear2.depthStencil.depth = 1;
                clear2.depthStencil.stencil = 0;

                var clearValues = new List<VkClearValue> { clear1, clear2 };
                renderPassInfo.clearValues = clearValues;

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
