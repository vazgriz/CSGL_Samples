using System;
using System.IO;
using System.Collections.Generic;

using CSGL;
using CSGL.GLFW;
using CSGL.GLFW.Unmanaged;
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

        CSGL.Vulkan.VkFormat swapchainImageFormat;
        VkExtent2D swapchainExtent;

        vkDestroySurfaceKHRDelegate destroySurface;
        vkCreateSwapchainKHRDelegate createSwapchain;
        vkGetPhysicalDeviceSurfaceSupportKHRDelegate getPhysicalDeviceSurfaceSupport;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHRDelegate getPhysicalDeviceSurfaceCapabilities;
        vkGetPhysicalDeviceSurfaceFormatsKHRDelegate getPhysicalDeviceSurfaceFormats;
        vkGetPhysicalDeviceSurfacePresentModesKHRDelegate getPhysicalDeviceSurfacePresentModes;
        vkDestroySwapchainKHRDelegate destroySwapchain;
        vkGetSwapchainImagesKHRDelegate getSwapchainImages;
        vkAcquireNextImageKHRDelegate acquireNextImage;
        vkQueuePresentKHRDelegate queuePresent;

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
            CreateDelegates();
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
            destroySwapchain(device, swapchain, alloc);
            VK.DestroyDevice(device, alloc);
            destroySurface(instance, surface, alloc);
            VK.DestroyInstance(instance, alloc);
            GLFW.Terminate();
        }

        void MainLoop() {
            var submitInfo = new VkSubmitInfo();
            submitInfo.sType = CSGL.Vulkan.VkStructureType.SubmitInfo;

            var waitSemaphores = new Native<VkSemaphore>(imageAvailableSemaphore);
            var waitStages = new Native<CSGL.Vulkan.VkPipelineStageFlags>(CSGL.Vulkan.VkPipelineStageFlags.ColorAttachmentOutputBit);
            var signalSemaphores = new Native<VkSemaphore>(renderFinishedSemaphore);
            var swapchains = new Native<VkSwapchainKHR>(swapchain);

            var commandBuffer = new Native<VkCommandBuffer>();
            var indexNative = new Native<uint>();

            submitInfo.waitSemaphoreCount = 1;
            submitInfo.pWaitSemaphores = waitSemaphores.Address;
            submitInfo.pWaitDstStageMask = waitStages.Address;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = commandBuffer.Address;
            submitInfo.signalSemaphoreCount = 1;
            submitInfo.pSignalSemaphores = signalSemaphores.Address;

            var submitInfoNative = new Native<VkSubmitInfo>(submitInfo);

            var presentInfo = new VkPresentInfoKHR();
            presentInfo.sType = CSGL.Vulkan.VkStructureType.PresentInfoKhr;
            presentInfo.waitSemaphoreCount = 1;
            presentInfo.pWaitSemaphores = signalSemaphores.Address;
            presentInfo.swapchainCount = 1;
            presentInfo.pSwapchains = swapchains.Address;
            presentInfo.pImageIndices = indexNative.Address;

            while (true) {
                GLFW.PollEvents();
                if (GLFW.WindowShouldClose(window)) break;

                if (reCreateSwapchainFlag) {
                    reCreateSwapchainFlag = false;
                    RecreateSwapchain();
                }

                uint imageIndex;
                var result = acquireNextImage(device, swapchain, ulong.MaxValue, imageAvailableSemaphore, VkFence.Null, out imageIndex);

                if (result == CSGL.Vulkan.VkResult.ErrorOutOfDateKhr || result == CSGL.Vulkan.VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                    continue;
                }

                commandBuffer.Value = commandBuffers[(int)imageIndex];
                swapchains.Value = swapchain;
                indexNative.Value = imageIndex;

                VK.QueueSubmit(graphicsQueue, 1, submitInfoNative.Address, VkFence.Null);
                result = queuePresent(presentQueue, ref presentInfo);

                if (result == CSGL.Vulkan.VkResult.ErrorOutOfDateKhr || result == CSGL.Vulkan.VkResult.SuboptimalKhr) {
                    RecreateSwapchain();
                }
            }

            VK.DeviceWaitIdle(device);
            waitSemaphores.Dispose();
            waitStages.Dispose();
            signalSemaphores.Dispose();
            swapchains.Dispose();
            commandBuffer.Dispose();
            submitInfoNative.Dispose();
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
            window = GLFW.CreateWindow(height, width, "Hello Triangle", MonitorPtr.Null, WindowPtr.Null);

            GLFW.SetWindowSizeCallback(window, OnWindowResized);
        }

        void OnWindowResized(WindowPtr window, int width, int height) {
            if (width == 0 || height == 0) return;
            reCreateSwapchainFlag = true;
        }

        void CreateInstance() {
            var appName = new InteropString("Hello Triangle");

            var appInfo = new VkApplicationInfo();
            appInfo.sType = CSGL.Vulkan.VkStructureType.ApplicationInfo;
            appInfo.pApplicationName = appName.Address;
            appInfo.applicationVersion = new CSGL.Vulkan.VkVersion(1, 0, 0);
            appInfo.engineVersion = new CSGL.Vulkan.VkVersion(0, 0, 1);
            appInfo.apiVersion = new CSGL.Vulkan.VkVersion(1, 0, 0);

            var appInfoNative = new Native<VkApplicationInfo>(appInfo);

            var info = new VkInstanceCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.InstanceCreateInfo;
            info.pApplicationInfo = appInfoNative.Address;

            var extensions = GLFW.GetRequiredInstanceExceptions();
            var extensionsNative = new NativeStringArray(extensions);
            info.ppEnabledExtensionNames = extensionsNative.Address;
            info.enabledExtensionCount = (uint)extensions.Count;

            var layersNative = new NativeStringArray(layers);
            info.ppEnabledLayerNames = layersNative.Address;
            info.enabledLayerCount = (uint)layers.Length;

            var result = VK.CreateInstance(ref info, alloc, out instance);

            appName.Dispose();
            appInfoNative.Dispose();
            extensionsNative.Dispose();
            layersNative.Dispose();
        }

        void CreateDelegates() {
            VK.Load(ref destroySurface, instance);
            VK.Load(ref getPhysicalDeviceSurfaceSupport, instance);
            VK.Load(ref getPhysicalDeviceSurfaceCapabilities, instance);
            VK.Load(ref getPhysicalDeviceSurfaceFormats, instance);
            VK.Load(ref getPhysicalDeviceSurfacePresentModes, instance);
            VK.Load(ref createSwapchain, instance);
            VK.Load(ref destroySwapchain, instance);
            VK.Load(ref getSwapchainImages, instance);
            VK.Load(ref acquireNextImage, instance);
            VK.Load(ref queuePresent, instance);
        }

        void CreateSurface() {
            var result = GLFW.CreateWindowSurface(instance.native, window, alloc, out surface.native);

        }

        void PickPhysicalDevice() {
            uint count = 0;
            VK.EnumeratePhysicalDevices(instance, ref count, IntPtr.Zero);
            var devices = new NativeArray<VkPhysicalDevice>(count);
            VK.EnumeratePhysicalDevices(instance, ref count, devices.Address);

            physicalDevice = devices[0];

            devices.Dispose();
        }

        void PickQueues() {
            uint count = 0;
            VK.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, ref count, IntPtr.Zero);
            var queues = new NativeArray<VkQueueFamilyProperties>(count);
            VK.GetPhysicalDeviceQueueFamilyProperties(physicalDevice, ref count, queues.Address);

            int g = -1;
            int p = -1;

            for (int i = 0; i < count; i++) {
                if (g == -1 && queues[i].queueCount > 0 && (queues[i].queueFlags & CSGL.Vulkan.VkQueueFlags.GraphicsBit) != 0) {
                    g = i;
                }

                bool support;
                getPhysicalDeviceSurfaceSupport(physicalDevice, (uint)i, surface, out support);
                if (p == -1 && queues[i].queueCount > 0 && support) {
                    p = i;
                }
            }

            graphicsIndex = (uint)g;
            presentIndex = (uint)p;

            queues.Dispose();
        }

        void CreateDevice() {
            var features = new Native<VkPhysicalDeviceFeatures>();
            VK.GetPhysicalDeviceFeatures(physicalDevice, features.Address);

            HashSet<uint> uniqueIndices = new HashSet<uint> { graphicsIndex, presentIndex };
            var queueInfos = new NativeArray<VkDeviceQueueCreateInfo>(uniqueIndices.Count);
            var priorities = new Native<float>(1);

            int i = 0;
            foreach (var ind in uniqueIndices) {
                var queueInfo = new VkDeviceQueueCreateInfo();
                queueInfo.sType = CSGL.Vulkan.VkStructureType.DeviceQueueCreateInfo;
                queueInfo.queueFamilyIndex = ind;
                queueInfo.queueCount = 1;

                queueInfo.pQueuePriorities = priorities.Address;

                queueInfos[i] = queueInfo;
                i++;
            }

            var info = new VkDeviceCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.DeviceCreateInfo;
            info.pQueueCreateInfos = queueInfos.Address;
            info.queueCreateInfoCount = (uint)uniqueIndices.Count;
            info.pEnabledFeatures = features.Address;

            var extensionsNative = new NativeStringArray(deviceExtensions);
            info.ppEnabledExtensionNames = extensionsNative.Address;
            info.enabledExtensionCount = (uint)deviceExtensions.Length;

            var result = VK.CreateDevice(physicalDevice, ref info, alloc, out device);

            VK.GetDeviceQueue(device, graphicsIndex, 0, out graphicsQueue);
            VK.GetDeviceQueue(device, presentIndex, 0, out presentQueue);

            features.Dispose();
            priorities.Dispose();
            queueInfos.Dispose();
            extensionsNative.Dispose();
        }

        SwapchainSupport GetSwapchainSupport(VkPhysicalDevice device) {
            var capNative = new Native<VkSurfaceCapabilitiesKHR>();
            getPhysicalDeviceSurfaceCapabilities(physicalDevice, surface, capNative.Address);

            uint count = 0;
            getPhysicalDeviceSurfaceFormats(physicalDevice, surface, ref count, IntPtr.Zero);
            var formatsNative = new NativeArray<VkSurfaceFormatKHR>(count);
            getPhysicalDeviceSurfaceFormats(physicalDevice, surface, ref count, formatsNative.Address);

            var formats = new List<VkSurfaceFormatKHR>((int)count);
            for (int i = 0; i < count; i++) {
                formats.Add(formatsNative[i]);
            }

            count = 0;
            getPhysicalDeviceSurfacePresentModes(physicalDevice, surface, ref count, IntPtr.Zero);
            var modesNative = new NativeArray<int>(count);
            getPhysicalDeviceSurfacePresentModes(physicalDevice, surface, ref count, modesNative.Address);

            var modes = new List<CSGL.Vulkan.VkPresentModeKHR>((int)count);
            for (int i = 0; i < count; i++) {
                modes.Add((CSGL.Vulkan.VkPresentModeKHR)modesNative[i]);
            }

            formatsNative.Dispose();
            modesNative.Dispose();

            return new SwapchainSupport(capNative, formats, modes);
        }

        VkSurfaceFormatKHR ChooseSwapSurfaceFormat(List<VkSurfaceFormatKHR> formats) {
            if (formats.Count == 1 && formats[0].format == CSGL.Vulkan.VkFormat.Undefined) {
                var result = new VkSurfaceFormatKHR();
                result.format = CSGL.Vulkan.VkFormat.B8G8R8A8_Unorm;
                result.colorSpace = CSGL.Vulkan.VkColorSpaceKHR.SrgbNonlinearKhr;
                return result;
            }

            foreach (var f in formats) {
                if (f.format == CSGL.Vulkan.VkFormat.B8G8R8A8_Unorm && f.colorSpace == CSGL.Vulkan.VkColorSpaceKHR.SrgbNonlinearKhr) {
                    return f;
                }
            }

            return formats[0];
        }

        CSGL.Vulkan.VkPresentModeKHR ChooseSwapPresentMode(List<CSGL.Vulkan.VkPresentModeKHR> modes) {
            foreach (var m in modes) {
                if (m == CSGL.Vulkan.VkPresentModeKHR.MailboxKhr) {
                    return m;
                }
            }

            return CSGL.Vulkan.VkPresentModeKHR.FifoKhr;
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
            info.sType = CSGL.Vulkan.VkStructureType.SwapchainCreateInfoKhr;
            info.surface = surface;
            info.minImageCount = imageCount;
            info.imageFormat = surfaceFormat.format;
            info.imageColorSpace = surfaceFormat.colorSpace;
            info.imageExtent = extent;
            info.imageArrayLayers = 1;
            info.imageUsage = CSGL.Vulkan.VkImageUsageFlags.ColorAttachmentBit;

            var queueFamilyIndices = new NativeArray<uint>(2);
            queueFamilyIndices[0] = graphicsIndex;
            queueFamilyIndices[1] = presentIndex;

            if (graphicsIndex != presentIndex) {
                info.imageSharingMode = CSGL.Vulkan.VkSharingMode.Concurrent;
                info.queueFamilyIndexCount = 2;
                info.pQueueFamilyIndices = queueFamilyIndices.Address;
            } else {
                info.imageSharingMode = CSGL.Vulkan.VkSharingMode.Exclusive;
            }

            info.preTransform = cap.currentTransform;
            info.compositeAlpha = CSGL.Vulkan.VkCompositeAlphaFlagsKHR.OpaqueBitKhr;
            info.presentMode = mode;
            info.clipped = 1;

            var oldSwapchain = swapchain;
            info.oldSwapchain = oldSwapchain;

            VkSwapchainKHR newSwapchain;
            var result = createSwapchain(device, ref info, alloc, out newSwapchain);

            if (swapchain != VkSwapchainKHR.Null) {
                destroySwapchain(device, swapchain, alloc);
            }
            swapchain = newSwapchain;

            getSwapchainImages(device, swapchain, ref imageCount, IntPtr.Zero);
            var swapchainImagesNative = new NativeArray<VkImage>(imageCount);
            getSwapchainImages(device, swapchain, ref imageCount, swapchainImagesNative.Address);

            swapchainImages = new List<VkImage>(swapchainImagesNative.Count);

            for (int i = 0; i < imageCount; i++) {
                swapchainImages.Add(swapchainImagesNative[i]);
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
                info.sType = CSGL.Vulkan.VkStructureType.ImageViewCreateInfo;
                info.image = image;
                info.viewType = CSGL.Vulkan.VkImageViewType._2D;
                info.format = swapchainImageFormat;
                info.components.r = CSGL.Vulkan.VkComponentSwizzle.Identity;
                info.components.g = CSGL.Vulkan.VkComponentSwizzle.Identity;
                info.components.b = CSGL.Vulkan.VkComponentSwizzle.Identity;
                info.components.a = CSGL.Vulkan.VkComponentSwizzle.Identity;
                info.subresourceRange.aspectMask = CSGL.Vulkan.VkImageAspectFlags.ColorBit;
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
            colorAttachment.samples = CSGL.Vulkan.VkSampleCountFlags._1_Bit;
            colorAttachment.loadOp = CSGL.Vulkan.VkAttachmentLoadOp.Clear;
            colorAttachment.storeOp = CSGL.Vulkan.VkAttachmentStoreOp.Store;
            colorAttachment.stencilLoadOp = CSGL.Vulkan.VkAttachmentLoadOp.DontCare;
            colorAttachment.stencilStoreOp = CSGL.Vulkan.VkAttachmentStoreOp.DontCare;
            colorAttachment.initialLayout = CSGL.Vulkan.VkImageLayout.Undefined;
            colorAttachment.finalLayout = CSGL.Vulkan.VkImageLayout.PresentSrcKhr;

            var colorAttachmentNative = new Native<VkAttachmentDescription>(colorAttachment);

            var colorAttachmentRef = new VkAttachmentReference();
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = CSGL.Vulkan.VkImageLayout.ColorAttachmentOptimal;

            var colorAttachmentRefNative = new Native<VkAttachmentReference>(colorAttachmentRef);

            var subpass = new VkSubpassDescription();
            subpass.pipelineBindPoint = CSGL.Vulkan.VkPipelineBindPoint.Graphics;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = colorAttachmentRefNative.Address;

            var subpassNative = new Native<VkSubpassDescription>(subpass);

            var dependency = new VkSubpassDependency();
            dependency.srcSubpass = uint.MaxValue;  //VK_SUBPASS_EXTERNAL
            dependency.dstSubpass = 0;
            dependency.srcStageMask = CSGL.Vulkan.VkPipelineStageFlags.BottomOfPipeBit;
            dependency.srcAccessMask = CSGL.Vulkan.VkAccessFlags.MemoryReadBit;
            dependency.dstStageMask = CSGL.Vulkan.VkPipelineStageFlags.ColorAttachmentOutputBit;
            dependency.dstAccessMask = CSGL.Vulkan.VkAccessFlags.ColorAttachmentReadBit
                                    | CSGL.Vulkan.VkAccessFlags.ColorAttachmentWriteBit;

            var dependencyNative = new Native<VkSubpassDependency>(dependency);

            var info = new VkRenderPassCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.RenderPassCreateInfo;
            info.attachmentCount = 1;
            info.pAttachments = colorAttachmentNative.Address;
            info.subpassCount = 1;
            info.pSubpasses = subpassNative.Address;
            info.dependencyCount = 1;
            info.pDependencies = dependencyNative.Address;

            if (renderPass != VkRenderPass.Null) {
                VK.DestroyRenderPass(device, renderPass, alloc);
            }

            var result = VK.CreateRenderPass(device, ref info, alloc, out renderPass);

            colorAttachmentNative.Dispose();
            colorAttachmentRefNative.Dispose();
            subpassNative.Dispose();
            dependencyNative.Dispose();
        }

        public VkShaderModule CreateShaderModule(byte[] code) {
            GCHandle handle = GCHandle.Alloc(code, GCHandleType.Pinned);

            var info = new VkShaderModuleCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.ShaderModuleCreateInfo;
            info.codeSize = (IntPtr)code.LongLength;
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
            vertInfo.sType = CSGL.Vulkan.VkStructureType.PipelineShaderStageCreateInfo;
            vertInfo.stage = CSGL.Vulkan.VkShaderStageFlags.VertexBit;
            vertInfo.module = vert;
            vertInfo.pName = entry.Address;

            var fragInfo = new VkPipelineShaderStageCreateInfo();
            fragInfo.sType = CSGL.Vulkan.VkStructureType.PipelineShaderStageCreateInfo;
            fragInfo.stage = CSGL.Vulkan.VkShaderStageFlags.FragmentBit;
            fragInfo.module = frag;
            fragInfo.pName = entry.Address;

            var shaderStages = new NativeArray<VkPipelineShaderStageCreateInfo>(2);
            shaderStages[0] = vertInfo;
            shaderStages[1] = fragInfo;

            var vertexInputInfo = new VkPipelineVertexInputStateCreateInfo();
            vertexInputInfo.sType = CSGL.Vulkan.VkStructureType.PipelineVertexInputStateCreateInfo;

            var vertexInputNative = new Native<VkPipelineVertexInputStateCreateInfo>(vertexInputInfo);

            var inputAssembly = new VkPipelineInputAssemblyStateCreateInfo();
            inputAssembly.sType = CSGL.Vulkan.VkStructureType.PipelineInputAssemblyStateCreateInfo;
            inputAssembly.topology = CSGL.Vulkan.VkPrimitiveTopology.TriangleList;

            var inputAssemblyNative = new Native<VkPipelineInputAssemblyStateCreateInfo>(inputAssembly);

            var viewport = new VkViewport();
            viewport.width = swapchainExtent.width;
            viewport.height = swapchainExtent.height;
            viewport.minDepth = 0f;
            viewport.maxDepth = 1f;

            var viewportNative = new Native<VkViewport>(viewport);

            var scissor = new VkRect2D();
            scissor.extent = swapchainExtent;

            var scissorNative = new Native<VkRect2D>(scissor);

            var viewportState = new VkPipelineViewportStateCreateInfo();
            viewportState.sType = CSGL.Vulkan.VkStructureType.PipelineViewportStateCreateInfo;
            viewportState.viewportCount = 1;
            viewportState.pViewports = viewportNative.Address;
            viewportState.scissorCount = 1;
            viewportState.pScissors = scissorNative.Address;

            var viewportStateNative = new Native<VkPipelineViewportStateCreateInfo>(viewportState);

            var rasterizer = new VkPipelineRasterizationStateCreateInfo();
            rasterizer.sType = CSGL.Vulkan.VkStructureType.PipelineRasterizationStateCreateInfo;
            rasterizer.polygonMode = CSGL.Vulkan.VkPolygonMode.Fill;
            rasterizer.lineWidth = 1f;
            rasterizer.cullMode = CSGL.Vulkan.VkCullModeFlags.BackBit;
            rasterizer.frontFace = CSGL.Vulkan.VkFrontFace.Clockwise;

            var rasterizerNative = new Native<VkPipelineRasterizationStateCreateInfo>(rasterizer);

            var multisampling = new VkPipelineMultisampleStateCreateInfo();
            multisampling.sType = CSGL.Vulkan.VkStructureType.PipelineMultisampleStateCreateInfo;
            multisampling.rasterizationSamples = CSGL.Vulkan.VkSampleCountFlags._1_Bit;
            multisampling.minSampleShading = 1f;

            var multisamplingNative = new Native<VkPipelineMultisampleStateCreateInfo>(multisampling);

            var colorBlendAttachment = new VkPipelineColorBlendAttachmentState();
            colorBlendAttachment.colorWriteMask = CSGL.Vulkan.VkColorComponentFlags.RBit
                                                | CSGL.Vulkan.VkColorComponentFlags.GBit
                                                | CSGL.Vulkan.VkColorComponentFlags.BBit
                                                | CSGL.Vulkan.VkColorComponentFlags.ABit;
            colorBlendAttachment.srcColorBlendFactor = CSGL.Vulkan.VkBlendFactor.One;
            colorBlendAttachment.dstColorBlendFactor = CSGL.Vulkan.VkBlendFactor.Zero;
            colorBlendAttachment.colorBlendOp = CSGL.Vulkan.VkBlendOp.Add;
            colorBlendAttachment.srcAlphaBlendFactor = CSGL.Vulkan.VkBlendFactor.One;
            colorBlendAttachment.dstAlphaBlendFactor = CSGL.Vulkan.VkBlendFactor.Zero;
            colorBlendAttachment.alphaBlendOp = CSGL.Vulkan.VkBlendOp.Add;

            var colorBlendAttachmentNative = new Native<VkPipelineColorBlendAttachmentState>(colorBlendAttachment);

            var colorBlending = new VkPipelineColorBlendStateCreateInfo();
            colorBlending.sType = CSGL.Vulkan.VkStructureType.PipelineColorBlendStateCreateInfo;
            colorBlending.logicOp = CSGL.Vulkan.VkLogicOp.Copy;
            colorBlending.attachmentCount = 1;
            colorBlending.pAttachments = colorBlendAttachmentNative.Address;

            var colorBlendingNative = new Native<VkPipelineColorBlendStateCreateInfo>(colorBlending);

            var pipelineLayoutInfo = new VkPipelineLayoutCreateInfo();
            pipelineLayoutInfo.sType = CSGL.Vulkan.VkStructureType.PipelineLayoutCreateInfo;

            if (pipelineLayout != VkPipelineLayout.Null) {
                VK.DestroyPipelineLayout(device, pipelineLayout, alloc);
            }
            var result = VK.CreatePipelineLayout(device, ref pipelineLayoutInfo, alloc, out pipelineLayout);

            var info = new VkGraphicsPipelineCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.GraphicsPipelineCreateInfo;
            info.stageCount = 2;
            info.pStages = shaderStages.Address;
            info.pVertexInputState = vertexInputNative.Address;
            info.pInputAssemblyState = inputAssemblyNative.Address;
            info.pViewportState = viewportStateNative.Address;
            info.pRasterizationState = rasterizerNative.Address;
            info.pMultisampleState = multisamplingNative.Address;
            info.pColorBlendState = colorBlendingNative.Address;
            info.layout = pipelineLayout;
            info.renderPass = renderPass;
            info.subpass = 0;
            info.basePipelineHandle = VkPipeline.Null;
            info.basePipelineIndex = -1;

            var infoNative = new Native<VkGraphicsPipelineCreateInfo>(info);
            var temp = new Native<VkPipeline>();

            if (pipeline != VkPipeline.Null) {
                VK.DestroyPipeline(device, pipeline, alloc);
            }

            result = VK.CreateGraphicsPipelines(device, VkPipelineCache.Null, 1, infoNative.Address, alloc, temp.Address);
            pipeline = temp.Value;

            infoNative.Dispose();
            temp.Dispose();

            entry.Dispose();
            shaderStages.Dispose();
            vertexInputNative.Dispose();
            inputAssemblyNative.Dispose();
            viewportNative.Dispose();
            scissorNative.Dispose();
            viewportStateNative.Dispose();
            rasterizerNative.Dispose();
            multisamplingNative.Dispose();
            colorBlendingNative.Dispose();
            colorBlendAttachmentNative.Dispose();
            VK.DestroyShaderModule(device, vert, alloc);
            VK.DestroyShaderModule(device, frag, alloc);
        }

        void CreateFramebuffers() {
            if (swapchainFramebuffers != null) {
                foreach (var fb in swapchainFramebuffers) VK.DestroyFramebuffer(device, fb, alloc);
            }

            swapchainFramebuffers = new List<VkFramebuffer>(swapchainImageViews.Count);

            for (int i = 0; i < swapchainImageViews.Count; i++) {
                var attachments = new Native<VkImageView>(swapchainImageViews[i]);

                var info = new VkFramebufferCreateInfo();
                info.sType = CSGL.Vulkan.VkStructureType.FramebufferCreateInfo;
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
            info.sType = CSGL.Vulkan.VkStructureType.CommandPoolCreateInfo;
            info.queueFamilyIndex = graphicsIndex;

            var result = VK.CreateCommandPool(device, ref info, alloc, out commandPool);
        }

        void CreateCommandBuffers() {
            if (commandBuffers != null) {
                var Native = new NativeArray<VkCommandBuffer>(commandBuffers);
                VK.FreeCommandBuffers(device, commandPool, (uint)commandBuffers.Count, Native.Address);
                Native.Dispose();
            }
            commandBuffers = new List<VkCommandBuffer>(swapchainFramebuffers.Count);

            var info = new VkCommandBufferAllocateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.CommandBufferAllocateInfo;
            info.commandPool = commandPool;
            info.level = CSGL.Vulkan.VkCommandBufferLevel.Primary;
            info.commandBufferCount = (uint)commandBuffers.Capacity;

            var commandBuffersNative = new NativeArray<VkCommandBuffer>(commandBuffers.Capacity);

            var result = VK.AllocateCommandBuffers(device, ref info, commandBuffersNative.Address);

            for (int i = 0; i < commandBuffers.Capacity; i++) {
                commandBuffers.Add(commandBuffersNative[i]);
            }
            commandBuffersNative.Dispose();

            for (int i = 0; i < commandBuffers.Count; i++) {
                var beginInfo = new VkCommandBufferBeginInfo();
                beginInfo.sType = CSGL.Vulkan.VkStructureType.CommandBufferBeginInfo;
                beginInfo.flags = CSGL.Vulkan.VkCommandBufferUsageFlags.SimultaneousUseBit;

                VK.BeginCommandBuffer(commandBuffers[i], ref beginInfo);

                var renderPassInfo = new VkRenderPassBeginInfo();
                renderPassInfo.sType = CSGL.Vulkan.VkStructureType.RenderPassBeginInfo;
                renderPassInfo.renderPass = renderPass;
                renderPassInfo.framebuffer = swapchainFramebuffers[i];
                renderPassInfo.renderArea.extent = swapchainExtent;

                VkClearValue clearColor = new VkClearValue();
                clearColor.color.float32_0 = 0;
                clearColor.color.float32_1 = 0;
                clearColor.color.float32_2 = 0;
                clearColor.color.float32_3 = 1f;

                var clearColorNative = new Native<VkClearValue>(clearColor);
                renderPassInfo.clearValueCount = 1;
                renderPassInfo.pClearValues = clearColorNative.Address;

                VK.CmdBeginRenderPass(commandBuffers[i], ref renderPassInfo, CSGL.Vulkan.VkSubpassContents.Inline);
                VK.CmdBindPipeline(commandBuffers[i], CSGL.Vulkan.VkPipelineBindPoint.Graphics, pipeline);
                VK.CmdDraw(commandBuffers[i], 3, 1, 0, 0);
                VK.CmdEndRenderPass(commandBuffers[i]);

                result = VK.EndCommandBuffer(commandBuffers[i]);

                clearColorNative.Dispose();
            }
        }

        void CreateSemaphores() {
            var info = new VkSemaphoreCreateInfo();
            info.sType = CSGL.Vulkan.VkStructureType.SemaphoreCreateInfo;

            VK.CreateSemaphore(device, ref info, alloc, out imageAvailableSemaphore);
            VK.CreateSemaphore(device, ref info, alloc, out renderFinishedSemaphore);
        }
    }

    struct SwapchainSupport {
        public Native<VkSurfaceCapabilitiesKHR> cap;
        public List<VkSurfaceFormatKHR> formats;
        public List<CSGL.Vulkan.VkPresentModeKHR> modes;

        public SwapchainSupport(Native<VkSurfaceCapabilitiesKHR> cap, List<VkSurfaceFormatKHR> formats, List<CSGL.Vulkan.VkPresentModeKHR> modes) {
            this.cap = cap;
            this.formats = formats;
            this.modes = modes;
        }
    }
}
