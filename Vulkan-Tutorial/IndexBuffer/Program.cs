using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;

using CSGL;
using CSGL.GLFW;
using CSGL.Vulkan;
using CSGL.GLFW.Unmanaged;

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
            result.stride = (int)Interop.SizeOf<Vertex>();
            result.inputRate = VkVertexInputRate.Vertex;

            return result;
        }

        public static List<VkVertexInputAttributeDescription> GetAttributeDescriptions() {
            Vertex v = new Vertex();
            var a = new VkVertexInputAttributeDescription();
            a.binding = 0;
            a.location = 0;
            a.format = VkFormat.R32G32B32_Sfloat;
            a.offset = (int)Interop.Offset(ref v, ref v.position);

            var b = new VkVertexInputAttributeDescription();
            b.binding = 0;
            b.location = 1;
            b.format = VkFormat.R32G32B32_Sfloat;
            b.offset = (int)Interop.Offset(ref v, ref v.color);

            return new List<VkVertexInputAttributeDescription> { a, b };
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

        int graphicsIndex;
        int presentIndex;
        VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;

        VkInstance instance;
        VkDebugReportCallback debugCallbacks;
        VkSurface surface;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkSwapchain swapchain;
        List<VkImage> swapchainImages;
        List<VkImageView> swapchainImageViews;
        VkRenderPass renderPass;
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        List<VkFramebuffer> swapchainFramebuffers;
        VkCommandPool commandPool;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
        List<VkCommandBuffer> commandBuffers;
        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;

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
            CreateGraphicsPipeline();
            CreateFramebuffers();
            CreateCommandPool();
            CreateVertexBuffer();
            CreateIndexBuffer();
            CreateCommandBuffers();
            CreateSemaphores();

            MainLoop();
        }

        public void Dispose() {
            imageAvailableSemaphore.Dispose();
            renderFinishedSemaphore.Dispose();
            indexBufferMemory.Dispose();
            indexBuffer.Dispose();
            vertexBufferMemory.Dispose();
            vertexBuffer.Dispose();
            commandPool.Dispose();
            foreach (var fb in swapchainFramebuffers) fb.Dispose();
            pipeline.Dispose();
            pipelineLayout.Dispose();
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
            var waitSemaphores = new List<VkSemaphore> { imageAvailableSemaphore };
            var waitStages = new List<VkPipelineStageFlags> { VkPipelineStageFlags.ColorAttachmentOutputBit };
            var signalSemaphores = new List<VkSemaphore> { renderFinishedSemaphore };
            var swapchains = new List<VkSwapchain> { swapchain };

            var commandBuffer = new List<VkCommandBuffer>() { null };
            var index = new List<int>() { 0 };

            var submitInfo = new VkSubmitInfo();
            submitInfo.waitSemaphores = waitSemaphores;
            submitInfo.waitDstStageMask = waitStages;
            submitInfo.commandBuffers = commandBuffer;
            submitInfo.signalSemaphores = signalSemaphores;

            var presentInfo = new VkPresentInfo();
            presentInfo.waitSemaphores = signalSemaphores;
            presentInfo.swapchains = swapchains;
            presentInfo.imageIndices = index;

            var submitInfos = new List<VkSubmitInfo> { submitInfo };

            GLFW.ShowWindow(window);

            while (true) {
                GLFW.PollEvents();
                if (GLFW.GetKey(window, CSGL.Input.KeyCode.Enter) == CSGL.Input.KeyAction.Press) {
                    break;
                }
                if (GLFW.WindowShouldClose(window)) break;

                if (recreateSwapchainFlag) {
                    recreateSwapchainFlag = false;
                    RecreateSwapchain();
                }

                int imageIndex;
                var result = swapchain.AcquireNextImage(-1, imageAvailableSemaphore, null, out imageIndex);

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
            window = GLFW.CreateWindow(width, height, "Index Buffer", MonitorPtr.Null, WindowPtr.Null);
        }

        void CreateInstance() {
            var extensions = new List<string>(GLFW.GetRequiredInstanceExceptions());
            foreach (var extension in instanceExtensions) {
                extensions.Add(extension);
            }

            var appInfo = new VkApplicationInfo {
                apiVersion = new VkVersion(1, 0, 0),
                applicationVersion = new VkVersion(1, 0, 0),
                engineVersion = new VkVersion(1, 0, 0),
                applicationName = "Index Buffer",
            };

            var info = new VkInstanceCreateInfo {
                applicationInfo = appInfo,
                extensions = extensions,
                layers = layers
            };
            instance = new VkInstance(info);
        }

        void DebugCallback(
            VkDebugReportFlagsEXT flags,
            VkDebugReportObjectTypeEXT objectType,
            long _object, long location,
            int messageCode, string layerPrefix, string message) {

            string type = flags.ToString();
            type = type.Substring(0, type.Length - 6);  //strip "BitExt"

            Console.WriteLine("[{0}] {1}", type, message);
        }

        void CreateDebugCallbacks() {
            var info = new VkDebugReportCallbackCreateInfo {
                callback = DebugCallback,
                flags = VkDebugReportFlagsEXT.DebugBitExt
                        | VkDebugReportFlagsEXT.ErrorBitExt
                        | VkDebugReportFlagsEXT.InformationBitExt
                        | VkDebugReportFlagsEXT.PerformanceWarningBitExt
                        | VkDebugReportFlagsEXT.WarningBitExt
            };

            debugCallbacks = new VkDebugReportCallback(instance, info);
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

            var uniqueIndices = new HashSet<int> { graphicsIndex, presentIndex };
            var priorities = new List<float> { 1f };
            var queueInfos = new List<VkDeviceQueueCreateInfo>(uniqueIndices.Count);

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
                extent.width = width;
                extent.height = height;

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

        void CreateImageViews() {
            if (swapchainImageViews != null) {
                foreach (var iv in swapchainImageViews) iv.Dispose();
            }

            swapchainImageViews = new List<VkImageView>();
            foreach (var image in swapchainImages) {
                var info = new VkImageViewCreateInfo();
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

                swapchainImageViews.Add(new VkImageView(device, info));
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
            rasterizer.frontFace = VkFrontFace.Clockwise;

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

            var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo();

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

            commandPool = new VkCommandPool(device, info);
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

        void CopyBuffer(VkBuffer src, VkBuffer dst, long size) {
            var info = new VkCommandBufferAllocateInfo();
            info.level = VkCommandBufferLevel.Primary;
            info.commandBufferCount = 1;

            var buffers = commandPool.Allocate(info);
            var buffer = buffers[0];

            var beginInfo = new VkCommandBufferBeginInfo();
            beginInfo.flags = VkCommandBufferUsageFlags.OneTimeSubmitBit;

            buffer.Begin(beginInfo);

            VkBufferCopy region = new VkBufferCopy();
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = size;

            buffer.CopyBuffer(src, dst, new VkBufferCopy[] { region });
            buffer.End();

            var submitInfo = new VkSubmitInfo();
            submitInfo.commandBuffers = new List<VkCommandBuffer> { buffer };

            graphicsQueue.Submit(new List<VkSubmitInfo> { submitInfo }, null);
            graphicsQueue.WaitIdle();

            commandPool.Free(buffers);
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

        void CreateCommandBuffers() {
            if (commandBuffers != null) {
                commandPool.Free(commandBuffers);
            }

            var info = new VkCommandBufferAllocateInfo();
            info.level = VkCommandBufferLevel.Primary;
            info.commandBufferCount = swapchainFramebuffers.Count;

            commandBuffers = new List<VkCommandBuffer>(commandPool.Allocate(info));

            for (int i = 0; i < commandBuffers.Count; i++) {
                var buffer = commandBuffers[i];
                var beginInfo = new VkCommandBufferBeginInfo();
                beginInfo.flags = VkCommandBufferUsageFlags.SimultaneousUseBit;

                buffer.Begin(beginInfo);

                var renderPassInfo = new VkRenderPassBeginInfo();
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapchainFramebuffers[i];
                renderPassInfo.renderArea.extent = swapchainExtent;

                VkClearValue clearColor = new VkClearValue {
                    color = new VkClearColorValue(0, 0, 0, 1f)
                };

                renderPassInfo.clearValues = new List<VkClearValue> { clearColor };

                buffer.BeginRenderPass(renderPassInfo, VkSubpassContents.Inline);
                buffer.BindPipeline(VkPipelineBindPoint.Graphics, pipeline);
                buffer.BindVertexBuffers(0, new VkBuffer[] { vertexBuffer }, new long[] { 0 });
                buffer.BindIndexBuffer(indexBuffer, 0, VkIndexType.UINT32);
                buffer.DrawIndexed(indices.Length, 1, 0, 0, 0);
                buffer.EndRenderPass();
                buffer.End();
            }
        }

        void CreateSemaphores() {
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
