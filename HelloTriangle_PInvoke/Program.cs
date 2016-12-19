using System;
using System.IO;
using System.Collections.Generic;

using CSGL;
using CSGL.GLFW;
using CSGL.Vulkan;
using CSGL.Vulkan.Unmanaged;
using System.Runtime.InteropServices;

namespace Samples {
    class Program : IDisposable {
        public static void Main(string[] args) {
            using (var p = new Program()) {
                p.Run();
            }
        }

        public Program() {
            GLFW.Init();
        }

        string[] layers = {
            "VK_LAYER_LUNARG_standard_validation",
            //"VK_LAYER_LUNARG_api_dump"
        };

        string[] deviceExtensions = {
            "VK_KHR_swapchain"
        };

        int height = 800;
        int width = 600;
        WindowPtr window;

        uint graphicsIndex;
        uint presentIndex;
        VkQueue graphicsQueue;
        VkQueue presentQueue;

        VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;

        IntPtr alloc = IntPtr.Zero;
        VkInstance instance;
        VkSurfaceKHR surface;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkSwapchainKHR swapchain;
        List<VkImage> swapchainImages;
        List<VkImageView> swapchainImageViews;
        VkRenderPass renderPass;
        VkPipelineLayout pipelineLayout;
        VkPipeline pipeline;
        List<VkFramebuffer> swapchainFramebuffers;
        VkCommandPool commandPool;
        List<VkCommandBuffer> commandBuffers;
        VkSemaphore imageAvailableSemaphore;
        VkSemaphore renderFinishedSemaphore;

        bool reCreateSwapchainFlag = false;

        void Run() {
            CreateWindow();
            CreateInstance();
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
            CreateCommandBuffers();
            CreateSemaphores();

            MainLoop();
        }

        public void Dispose() {
            VK.DestroySemaphore(device, imageAvailableSemaphore, alloc);
            VK.DestroySemaphore(device, renderFinishedSemaphore, alloc);
            VK.DestroyCommandPool(device, commandPool, alloc);
            foreach (var fb in swapchainFramebuffers) VK.DestroyFramebuffer(device, fb, alloc);
            VK.DestroyPipeline(device, pipeline, alloc);
            VK.DestroyPipelineLayout(device, pipelineLayout, alloc);
            VK.DestroyRenderPass(device, renderPass, alloc);
            foreach (var iv in swapchainImageViews) VK.DestroyImageView(device, iv, alloc);
            VK.DestroySwapchainKHR(device, swapchain, alloc);
            VK.DestroyDevice(device, alloc);
            VK.DestroySurfaceKHR(instance, surface, alloc);
            VK.DestroyInstance(instance, alloc);
            GLFW.Terminate();
        }

        void MainLoop() {
            var submitInfo = new VkSubmitInfo();
            submitInfo.sType = VkStructureType.SubmitInfo;

            var waitSemaphores = new Marshalled<VkSemaphore>(imageAvailableSemaphore);
            var waitStages = new Marshalled<uint>((uint)VkPipelineStageFlags.ColorAttachmentOutputBit);
            var signalSemaphores = new Marshalled<VkSemaphore>(renderFinishedSemaphore);
            var swapchains = new Marshalled<VkSwapchainKHR>(swapchain);

            var commandBuffer = new Marshalled<VkCommandBuffer>();
            var indexMarshalled = new Marshalled<uint>();

            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores.Address;
            submitInfo.pWaitDstStageMask = waitStages.Address;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = commandBuffer.Address;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores.Address;

            var submitInfoMarshalled = new Marshalled<VkSubmitInfo>(submitInfo);

            var presentInfo = new VkPresentInfoKHR();
            presentInfo.sType = VkStructureType.PresentInfoKhr;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores.Address;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains.Address;
            presentInfo.pImageIndices = indexMarshalled.Address;

            while (true) {
                GLFW.PollEvents();
                if (GLFW.WindowShouldClose(window)) break;

                if (reCreateSwapchainFlag) {
                    reCreateSwapchainFlag = false;
                    RecreateSwapchain();
                }

                uint imageIndex;
                var result = VK.AcquireNextImageKHR(device, swapchain, ulong.MaxValue, imageAvailableSemaphore, VkFence.Null, out imageIndex);

                if (result == VkResult.ErrorOutOfDateKhr || result == VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                    continue;
                }

                commandBuffer.Value = commandBuffers[(int)imageIndex];
                swapchains.Value = swapchain;
                indexMarshalled.Value = imageIndex;

                VK.QueueSubmit(graphicsQueue, 1, submitInfoMarshalled.Address, VkFence.Null);
                result = VK.QueuePresentKHR(presentQueue, ref presentInfo);

                if (result == VkResult.ErrorOutOfDateKhr || result == VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                }
            }

            VK.DeviceWaitIdle(device);
            waitSemaphores.Dispose();
            waitStages.Dispose();
            signalSemaphores.Dispose();
            swapchains.Dispose();
            commandBuffer.Dispose();
            submitInfoMarshalled.Dispose();
        }

        void RecreateSwapchain() {
            VK.DeviceWaitIdle(device);
            CreateSwapchain();
            CreateImageViews();
            CreateRenderPass();
            CreateGraphicsPipeline();
            CreateFramebuffers();
            CreateCommandBuffers();
        }

        void CreateWindow() {
            GLFW.WindowHint(WindowHint.ClientAPI, (int)ClientAPI.NoAPI);
            window = GLFW.CreateWindow(height, width, "Vulkan Test", MonitorPtr.Null, WindowPtr.Null);

            GLFW.SetWindowSizeCallback(window, OnWindowResized);
        }

        void OnWindowResized(WindowPtr window, int width, int height) {
            if (width == 0 || height == 0) return;
            reCreateSwapchainFlag = true;
        }

        void CreateInstance() {
            var appName = new InteropString("Hello Triangle");

            var appInfo = new VkApplicationInfo();
            appInfo.sType = VkStructureType.ApplicationInfo;
            appInfo.pApplicationName = appName.Address;
            appInfo.applicationVersion = new VkVersion(1, 0, 0);
            appInfo.engineVersion = new VkVersion(0, 0, 1);
            appInfo.apiVersion = new VkVersion(1, 0, 0);

            var appInfoMarshalled = new Marshalled<VkApplicationInfo>(appInfo);

            var info = new VkInstanceCreateInfo();
            info.sType = VkStructureType.InstanceCreateInfo;
            info.pApplicationInfo = appInfoMarshalled.Address;

            var extensions = GLFW.GetRequiredInstanceExceptions();
            var extensionsMarshalled = new NativeStringArray(extensions);
            info.ppEnabledExtensionNames = extensionsMarshalled.Address;
            info.enabledExtensionCount = (uint)extensions.Length;

            var layersMarshalled = new NativeStringArray(layers);
            info.ppEnabledLayerNames = layersMarshalled.Address;
            info.enabledLayerCount = (uint)layers.Length;

            var result = VK.CreateInstance(ref info, alloc, out instance);

            appName.Dispose();
            appInfoMarshalled.Dispose();
            extensionsMarshalled.Dispose();
            layersMarshalled.Dispose();
        }

        void CreateSurface() {
            var result = GLFW.CreateWindowSurface(instance, window, alloc, out surface);

        }

        void PickPhysicalDevice() {
            uint count = 0;
            VK.EnumeratePhysicalDevices(instance, ref count, IntPtr.Zero);
            var devices = new MarshalledArray<VkPhysicalDevice>(count);
            VK.EnumeratePhysicalDevices(instance, ref count, devices.Address);

            physicalDevice = devices[0];

            devices.Dispose();
        }

        void PickQueues() {
            uint count = 0;
            VK.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, ref count, IntPtr.Zero);
            var queues = new MarshalledArray<VkQueueFamilyProperties>(count);
            VK.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, ref count, queues.Address);

            int g = -1;
            int p = -1;

            for (int i = 0; i < count; i++) {
                if (queues.Count > 0 && (queues[i].queueFlags & VkQueueFlags.GraphicsBit) != 0) {
                    g = i;
                }

                bool support = false;
                VK.GetPhysicalDeviceSurfaceSupportKHR(physicalDevice, (uint)i, surface, out support);
                if (queues.Count > 0 && support) {
                    p = i;
                }
            }

            graphicsIndex = (uint)g;
            presentIndex = (uint)p;

            queues.Dispose();
        }

        void CreateDevice() {
            var features = new Marshalled<VkPhysicalDeviceFeatures>();
            VK.GetPhysicalDeviceFeatures(physicalDevice, features.Address);

            HashSet<uint> uniqueIndices = new HashSet<uint> { graphicsIndex, presentIndex };
            var queueInfos = new MarshalledArray<VkDeviceQueueCreateInfo>(uniqueIndices.Count);
            var priorities = new Marshalled<float>(1);

            int i = 0;
            foreach (var ind in uniqueIndices) {
                var queueInfo = new VkDeviceQueueCreateInfo();
                queueInfo.sType = VkStructureType.DeviceQueueCreateInfo;
                queueInfo.queueFamilyIndex = ind;
                queueInfo.queueCount = 1;

                queueInfo.pQueuePriorities = priorities.Address;

                queueInfos[i] = queueInfo;
                i++;
            }

            var info = new VkDeviceCreateInfo();
            info.sType = VkStructureType.DeviceCreateInfo;
            info.pQueueCreateInfos = queueInfos.Address;
            info.queueCreateInfoCount = 1;
            info.pEnabledFeatures = features.Address;

            var extensionsMarshalled = new NativeStringArray(deviceExtensions);
            info.ppEnabledExtensionNames = extensionsMarshalled.Address;
            info.enabledExtensionCount = (uint)deviceExtensions.Length;

            var result = VK.CreateDevice(physicalDevice, ref info, alloc, out device);

            VK.GetDeviceQueue(device, graphicsIndex, 0, out graphicsQueue);
            VK.GetDeviceQueue(device, presentIndex, 0, out presentQueue);

            features.Dispose();
            priorities.Dispose();
            queueInfos.Dispose();
            extensionsMarshalled.Dispose();
        }

        SwapchainSupport GetSwapchainSupport(VkPhysicalDevice device) {
            var capMarshalled = new Marshalled<VkSurfaceCapabilitiesKHR>();
            VK.GetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, capMarshalled.Address);

            uint count = 0;
            VK.GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, ref count, IntPtr.Zero);
            var formatsMarshalled = new MarshalledArray<VkSurfaceFormatKHR>(count);
            VK.GetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, ref count, formatsMarshalled.Address);

            var formats = new List<VkSurfaceFormatKHR>((int)count);
            for (int i = 0; i < count; i++) {
                formats.Add(formatsMarshalled[i]);
            }

            count = 0;
            VK.GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, ref count, IntPtr.Zero);
            var modesMarshalled = new MarshalledArray<int>(count);
            VK.GetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, ref count, modesMarshalled.Address);

            var modes = new List<VkPresentModeKHR>((int)count);
            for (int i = 0; i < count; i++) {
                modes.Add((VkPresentModeKHR)modesMarshalled[i]);
            }

            formatsMarshalled.Dispose();
            modesMarshalled.Dispose();

            return new SwapchainSupport(capMarshalled, formats, modes);
        }

        VkSurfaceFormatKHR ChooseSwapSurfaceFormat(List<VkSurfaceFormatKHR> formats) {
            if (formats.Count == 1 && formats[0].format == VkFormat.Undefined) {
                var result = new VkSurfaceFormatKHR();
                result.format = VkFormat.B8g8r8a8Unorm;
                result.colorSpace = VkColorSpaceKHR.SrgbNonlinearKhr;
                return result;
            }

            foreach (var f in formats) {
                if (f.format == VkFormat.B8g8r8a8Unorm && f.colorSpace == VkColorSpaceKHR.SrgbNonlinearKhr) {
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
            var cap = support.cap.Value;

            var surfaceFormat = ChooseSwapSurfaceFormat(support.formats);
            var mode = ChooseSwapPresentMode(support.modes);
            var extent = ChooseSwapExtent(ref cap);

            uint imageCount = cap.minImageCount + 1;
            if (cap.maxImageCount > 0 && imageCount > cap.maxImageCount) {
                imageCount = cap.maxImageCount;
            }

            var info = new VkSwapchainCreateInfoKHR();
            info.sType = VkStructureType.SwapchainCreateInfoKhr;
            info.surface = surface;
            info.minImageCount = imageCount;
            info.imageFormat = surfaceFormat.format;
            info.imageColorSpace = surfaceFormat.colorSpace;
            info.imageExtent = extent;
            info.imageArrayLayers = 1;
            info.imageUsage = VkImageUsageFlags.ColorAttachmentBit;

            var queueFamilyIndices = new MarshalledArray<uint>(2);
            queueFamilyIndices[0] = graphicsIndex;
            queueFamilyIndices[1] = presentIndex;

            if (graphicsIndex != presentIndex) {
                info.imageSharingMode = VkSharingMode.Concurrent;
                info.queueFamilyIndexCount = 2;
                info.pQueueFamilyIndices = queueFamilyIndices.Address;
            } else {
                info.imageSharingMode = VkSharingMode.Exclusive;
            }

            info.preTransform = cap.currentTransform;
            info.compositeAlpha = VkCompositeAlphaFlagsKHR.OpaqueBitKhr;
            info.presentMode = mode;
            info.clipped = 1;

            var oldSwapchain = swapchain;
            info.oldSwapchain = oldSwapchain;

            VkSwapchainKHR newSwapchain;
            var result = VK.CreateSwapchainKHR(device, ref info, alloc, out newSwapchain);

            if (swapchain != VkSwapchainKHR.Null) {
                VK.DestroySwapchainKHR(device, swapchain, alloc);
            }
            swapchain = newSwapchain;

            VK.GetSwapchainImagesKHR(device, swapchain, ref imageCount, IntPtr.Zero);
            var swapchainImagesMarshalled = new MarshalledArray<VkImage>(imageCount);
            VK.GetSwapchainImagesKHR(device, swapchain, ref imageCount, swapchainImagesMarshalled.Address);

            swapchainImages = new List<VkImage>(swapchainImagesMarshalled.Count);

            for (int i = 0; i < imageCount; i++) {
                swapchainImages.Add(swapchainImagesMarshalled[i]);
            }

            swapchainImageFormat = surfaceFormat.format;
            swapchainExtent = extent;

            support.cap.Dispose();
            queueFamilyIndices.Dispose();
        }

        void CreateImageViews() {
            if (swapchainImageViews != null) {
                foreach (var iv in swapchainImageViews) VK.DestroyImageView(device, iv, alloc);
            }
            swapchainImageViews = new List<VkImageView>(swapchainImages.Count);

            foreach (var image in swapchainImages) {
                var info = new VkImageViewCreateInfo();
                info.sType = VkStructureType.ImageViewCreateInfo;
                info.image = image;
                info.viewType = VkImageViewType._2d;
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

                VkImageView temp;
                var result = VK.CreateImageView(device, ref info, alloc, out temp);
                swapchainImageViews.Add(temp);
            }
        }

        void CreateRenderPass() {
            var colorAttachment = new VkAttachmentDescription();
            colorAttachment.format = swapchainImageFormat;
            colorAttachment.samples = VkSampleCountFlags._1Bit;
            colorAttachment.loadOp = VkAttachmentLoadOp.Clear;
            colorAttachment.storeOp = VkAttachmentStoreOp.Store;
            colorAttachment.stencilLoadOp = VkAttachmentLoadOp.DontCare;
            colorAttachment.stencilStoreOp = VkAttachmentStoreOp.DontCare;
            colorAttachment.initialLayout = VkImageLayout.Undefined;
            colorAttachment.finalLayout = VkImageLayout.PresentSrcKhr;

            var colorAttachmentMarshalled = new Marshalled<VkAttachmentDescription>(colorAttachment);

            var colorAttachmentRef = new VkAttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VkImageLayout.ColorAttachmentOptimal;

            var colorAttachmentRefMarshalled = new Marshalled<VkAttachmentReference>(colorAttachmentRef);

            var subpass = new VkSubpassDescription();
            subpass.pipelineBindPoint = VkPipelineBindPoint.Graphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = colorAttachmentRefMarshalled.Address;

            var subpassMarshalled = new Marshalled<VkSubpassDescription>(subpass);

            var dependency = new VkSubpassDependency();
            dependency.srcSubpass = uint.MaxValue;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = VkPipelineStageFlags.BottomOfPipeBit;
            dependency.srcAccessMask = VkAccessFlags.MemoryReadBit;
            dependency.dstStageMask = VkPipelineStageFlags.ColorAttachmentOutputBit;
            dependency.dstAccessMask = VkAccessFlags.ColorAttachmentReadBit
                                    | VkAccessFlags.ColorAttachmentWriteBit;

            var dependencyMarshalled = new Marshalled<VkSubpassDependency>(dependency);

            var info = new VkRenderPassCreateInfo();
            info.sType = VkStructureType.RenderPassCreateInfo;
            info.attachmentCount = 1;
            info.pAttachments = colorAttachmentMarshalled.Address;
            info.subpassCount = 1;
            info.pSubpasses = subpassMarshalled.Address;
            info.dependencyCount = 1;
            info.pDependencies = dependencyMarshalled.Address;

            if (renderPass != VkRenderPass.Null) {
                VK.DestroyRenderPass(device, renderPass, alloc);
            }

            var result = VK.CreateRenderPass(device, ref info, alloc, out renderPass);

            colorAttachmentMarshalled.Dispose();
            colorAttachmentRefMarshalled.Dispose();
            subpassMarshalled.Dispose();
            dependencyMarshalled.Dispose();
        }

        public VkShaderModule CreateShaderModule(byte[] code) {
            GCHandle handle = GCHandle.Alloc(code, GCHandleType.Pinned);

            var info = new VkShaderModuleCreateInfo();
            info.sType = VkStructureType.ShaderModuleCreateInfo;
            info.codeSize = (ulong)code.LongLength;
            info.pCode = handle.AddrOfPinnedObject();

            VkShaderModule temp;

            var result = VK.CreateShaderModule(device, ref info, alloc, out temp);
            handle.Free();

            return temp;
        }

        void CreateGraphicsPipeline() {
            VkShaderModule vert = CreateShaderModule(File.ReadAllBytes("vert.spv"));
            VkShaderModule frag = CreateShaderModule(File.ReadAllBytes("frag.spv"));

            InteropString entry = new InteropString("main");

            var vertInfo = new VkPipelineShaderStageCreateInfo();
            vertInfo.sType = VkStructureType.PipelineShaderStageCreateInfo;
            vertInfo.stage = VkShaderStageFlags.VertexBit;
            vertInfo.module = vert;
            vertInfo.pName = entry.Address;

            var fragInfo = new VkPipelineShaderStageCreateInfo();
            fragInfo.sType = VkStructureType.PipelineShaderStageCreateInfo;
            fragInfo.stage = VkShaderStageFlags.FragmentBit;
            fragInfo.module = frag;
            fragInfo.pName = entry.Address;

            var shaderStages = new MarshalledArray<VkPipelineShaderStageCreateInfo>(2);
            shaderStages[0] = vertInfo;
            shaderStages[1] = fragInfo;

            var vertexInputInfo = new VkPipelineVertexInputStateCreateInfo();
            vertexInputInfo.sType = VkStructureType.PipelineVertexInputStateCreateInfo;

            var vertexInputMarshalled = new Marshalled<VkPipelineVertexInputStateCreateInfo>(vertexInputInfo);

            var inputAssembly = new VkPipelineInputAssemblyStateCreateInfo();
            inputAssembly.sType = VkStructureType.PipelineInputAssemblyStateCreateInfo;
            inputAssembly.topology = VkPrimitiveTopology.TriangleList;

            var inputAssemblyMarshalled = new Marshalled<VkPipelineInputAssemblyStateCreateInfo>(inputAssembly);

            var viewport = new VkViewport();
            viewport.width = swapchainExtent.width;
            viewport.height = swapchainExtent.height;
            viewport.minDepth = 0f;
            viewport.maxDepth = 1f;

            var viewportMarshalled = new Marshalled<VkViewport>(viewport);

            var scissor = new VkRect2D();
            scissor.extent = swapchainExtent;

            var scissorMarshalled = new Marshalled<VkRect2D>(scissor);

            var viewportState = new VkPipelineViewportStateCreateInfo();
            viewportState.sType = VkStructureType.PipelineViewportStateCreateInfo;
            viewportState.viewportCount = 1;
            viewportState.pViewports = viewportMarshalled.Address;
            viewportState.scissorCount = 1;
            viewportState.pScissors = scissorMarshalled.Address;

            var viewportStateMarshalled = new Marshalled<VkPipelineViewportStateCreateInfo>(viewportState);

            var rasterizer = new VkPipelineRasterizationStateCreateInfo();
            rasterizer.sType = VkStructureType.PipelineRasterizationStateCreateInfo;
            rasterizer.polygonMode = VkPolygonMode.Fill;
            rasterizer.lineWidth = 1f;
            rasterizer.cullMode = VkCullModeFlags.BackBit;
            rasterizer.frontFace = VkFrontFace.Clockwise;

            var rasterizerMarshalled = new Marshalled<VkPipelineRasterizationStateCreateInfo>(rasterizer);

            var multisampling = new VkPipelineMultisampleStateCreateInfo();
            multisampling.sType = VkStructureType.PipelineMultisampleStateCreateInfo;
            multisampling.rasterizationSamples = VkSampleCountFlags._1Bit;
            multisampling.minSampleShading = 1f;

            var multisamplingMarshalled = new Marshalled<VkPipelineMultisampleStateCreateInfo>(multisampling);

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

            var colorBlendAttachmentMarshalled = new Marshalled<VkPipelineColorBlendAttachmentState>(colorBlendAttachment);

            var colorBlending = new VkPipelineColorBlendStateCreateInfo();
            colorBlending.sType = VkStructureType.PipelineColorBlendStateCreateInfo;
            colorBlending.logicOp = VkLogicOp.Copy;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = colorBlendAttachmentMarshalled.Address;

            var colorBlendingMarshalled = new Marshalled<VkPipelineColorBlendStateCreateInfo>(colorBlending);

            var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo();
            pipelineLayoutInfo.sType = VkStructureType.PipelineLayoutCreateInfo;

            if (pipelineLayout != VkPipelineLayout.Null) {
                VK.DestroyPipelineLayout(device, pipelineLayout, alloc);
            }
            var result = VK.CreatePipelineLayout(device, ref pipelineLayoutInfo, alloc, out pipelineLayout);

            var info = new VkGraphicsPipelineCreateInfo();
            info.sType = VkStructureType.GraphicsPipelineCreateInfo;
            info.stageCount = 2;
            info.pStages = shaderStages.Address;
            info.pVertexInputState = vertexInputMarshalled.Address;
            info.pInputAssemblyState = inputAssemblyMarshalled.Address;
            info.pViewportState = viewportStateMarshalled.Address;
            info.pRasterizationState = rasterizerMarshalled.Address;
            info.pMultisampleState = multisamplingMarshalled.Address;
            info.pColorBlendState = colorBlendingMarshalled.Address;
            info.layout = pipelineLayout;
            info.renderPass = renderPass;
            info.subpass = 0;
            info.basePipelineHandle = VkPipeline.Null;
            info.basePipelineIndex = -1;

            var infoMarshalled = new Marshalled<VkGraphicsPipelineCreateInfo>(info);
            var temp = new Marshalled<VkPipeline>();

            if (pipeline != VkPipeline.Null) {
                VK.DestroyPipeline(device, pipeline, alloc);
            }

            result = VK.CreateGraphicsPipelines(device, VkPipelineCache.Null, 1, infoMarshalled.Address, alloc, temp.Address);
            pipeline = temp.Value;

            infoMarshalled.Dispose();
            temp.Dispose();

            entry.Dispose();
            shaderStages.Dispose();
            vertexInputMarshalled.Dispose();
            inputAssemblyMarshalled.Dispose();
            viewportMarshalled.Dispose();
            scissorMarshalled.Dispose();
            viewportStateMarshalled.Dispose();
            rasterizerMarshalled.Dispose();
            multisamplingMarshalled.Dispose();
            colorBlendingMarshalled.Dispose();
            colorBlendAttachmentMarshalled.Dispose();
            VK.DestroyShaderModule(device, vert, alloc);
            VK.DestroyShaderModule(device, frag, alloc);
        }

        void CreateFramebuffers() {
            if (swapchainFramebuffers != null) {
                foreach (var fb in swapchainFramebuffers) VK.DestroyFramebuffer(device, fb, alloc);
            }

            swapchainFramebuffers = new List<VkFramebuffer>(swapchainImageViews.Count);

            for (int i = 0; i < swapchainImageViews.Count; i++) {
                var attachments = new Marshalled<VkImageView>(swapchainImageViews[i]);

                var info = new VkFramebufferCreateInfo();
                info.sType = VkStructureType.FramebufferCreateInfo;
                info.renderPass = renderPass;
                info.attachmentCount = 1;
                info.pAttachments = attachments.Address;
                info.width = swapchainExtent.width;
                info.height = swapchainExtent.height;
                info.layers = 1;

                VkFramebuffer temp;

                var result = VK.CreateFramebuffer(device, ref info, alloc, out temp);
                swapchainFramebuffers.Add(temp);

                attachments.Dispose();
            }
        }

        void CreateCommandPool() {
            var info = new VkCommandPoolCreateInfo();
            info.sType = VkStructureType.CommandPoolCreateInfo;
            info.queueFamilyIndex = graphicsIndex;

            var result = VK.CreateCommandPool(device, ref info, alloc, out commandPool);
        }

        void CreateCommandBuffers() {
            if (commandBuffers != null) {
                var marshalled = new MarshalledArray<VkCommandBuffer>(commandBuffers);
                VK.FreeCommandBuffers(device, commandPool, (uint)commandBuffers.Count, marshalled.Address);
                marshalled.Dispose();
            }
            commandBuffers = new List<VkCommandBuffer>(swapchainFramebuffers.Count);

            var info = new VkCommandBufferAllocateInfo();
            info.sType = VkStructureType.CommandBufferAllocateInfo;
            info.commandPool = commandPool;
            info.level = VkCommandBufferLevel.Primary;
            info.commandBufferCount = (uint)commandBuffers.Capacity;

            var commandBuffersMarshalled = new MarshalledArray<VkCommandBuffer>(commandBuffers.Capacity);

            var result = VK.AllocateCommandBuffers(device, ref info, commandBuffersMarshalled.Address);

            for (int i = 0; i < commandBuffers.Capacity; i++) {
                commandBuffers.Add(commandBuffersMarshalled[i]);
            }
            commandBuffersMarshalled.Dispose();

            for (int i = 0; i < commandBuffers.Count; i++) {
                var beginInfo = new VkCommandBufferBeginInfo();
                beginInfo.sType = VkStructureType.CommandBufferBeginInfo;
                beginInfo.flags = VkCommandBufferUsageFlags.SimultaneousUseBit;

                VK.BeginCommandBuffer(commandBuffers[i], ref beginInfo);

                var renderPassInfo = new VkRenderPassBeginInfo();
                renderPassInfo.sType = VkStructureType.RenderPassBeginInfo;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapchainFramebuffers[i];
                renderPassInfo.renderArea.extent = swapchainExtent;

                VkClearValue clearColor = new VkClearValue();
                clearColor.color.float32_0 = 0;
                clearColor.color.float32_1 = 0;
                clearColor.color.float32_2 = 0;
                clearColor.color.float32_3 = 1f;

                var clearColorMarshalled = new Marshalled<VkClearValue>(clearColor);
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = clearColorMarshalled.Address;

                VK.CmdBeginRenderPass(commandBuffers[i], ref renderPassInfo, VkSubpassContents.Inline);
                VK.CmdBindPipeline(commandBuffers[i], VkPipelineBindPoint.Graphics, pipeline);
                VK.CmdDraw(commandBuffers[i], 3, 1, 0, 0);
                VK.CmdEndRenderPass(commandBuffers[i]);

                result = VK.EndCommandBuffer(commandBuffers[i]);

                clearColorMarshalled.Dispose();
            }
        }

        void CreateSemaphores() {
            var info = new VkSemaphoreCreateInfo();
            info.sType = VkStructureType.SemaphoreCreateInfo;

            VK.CreateSemaphore(device, ref info, alloc, out imageAvailableSemaphore);
            VK.CreateSemaphore(device, ref info, alloc, out renderFinishedSemaphore);
        }
    }

    struct SwapchainSupport {
        public Marshalled<VkSurfaceCapabilitiesKHR> cap;
        public List<VkSurfaceFormatKHR> formats;
        public List<VkPresentModeKHR> modes;

        public SwapchainSupport(Marshalled<VkSurfaceCapabilitiesKHR> cap, List<VkSurfaceFormatKHR> formats, List<VkPresentModeKHR> modes) {
            this.cap = cap;
            this.formats = formats;
            this.modes = modes;
        }
    }
}
