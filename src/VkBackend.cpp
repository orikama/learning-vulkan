#include "VkBackend.hpp"

#include "Window.hpp"

#include <stdexcept> // std::runtime_error
#include <algorithm>
#include <optional>
#include <cstring> // std::strcmp
#include <limits>
#include <unordered_set>
#include <vector>
#include <array>
#include <filesystem>
#include <fstream>


// TODO: Remove globals
const char* kShaderVertexPath = "shader.vspv";
const char* kShaderFragmentPath = "shader.fspv";

#ifdef NDEBUG
    constexpr bool kEnableValidationLayers = false;
#else
    constexpr bool kEnableValidationLayers = true;
#endif

const std::vector<const char*> kValidationLayers{ "VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor" };
const std::vector<const char*> kDeviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };


// NOTE: Can be removed?
struct QueueFamilyIndices
{
    std::optional<ui32> graphicsFamily;
    std::optional<ui32> presentFamily;

    bool isComplete()
    {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapchainSupportDetails
{
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
    vk::SurfaceCapabilitiesKHR capabilities;
};


auto _checkAPIVersionSupport(const ui32 requestedVersion)   -> void;
auto _getRequiredExtensions()                               -> std::vector<const char*>;
auto _checkValidationLayersSupport()                        -> bool;
auto _makeDebugUtilsMessengerCreateInfo()                   -> vk::DebugUtilsMessengerCreateInfoEXT;

auto _isDeviceSuitable(const vk::PhysicalDevice device,
                       const vk::UniqueSurfaceKHR& surface)                 -> bool;
auto _getRequiredQueueFamilies(const vk::PhysicalDevice device,
                               const vk::UniqueSurfaceKHR& surface)         -> QueueFamilyIndices;
auto _checkPhysicalDeviceExtensionSupport(const vk::PhysicalDevice device)  -> bool;

auto _querySwapchainSupport(const vk::PhysicalDevice device,
                            const vk::UniqueSurfaceKHR& surface)            -> SwapchainSupportDetails;
auto _chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)    -> vk::SurfaceFormatKHR;
auto _choosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)   -> vk::PresentModeKHR;
auto _chooseSurfaceExtent(const vk::SurfaceCapabilitiesKHR& capabilities, ui32 width, ui32 height) -> vk::Extent2D;

auto _readShaderFile(const std::string_view shaderPath)         -> std::vector<char>;
auto _createShaderModule(const std::vector<char>& shaderCode,
                         const vk::UniqueDevice& device)        -> vk::UniqueShaderModule;


namespace vulkan
{

void VkBackend::Init(const Window& window)
{
    _CreateVkInstance(VK_API_VERSION_1_2);
    _SetupDebugMessenger();
    _CreateSurface(window.GetWindowHandle());
    _SelectPhysicalDevice();
    _CreateLogicalDeviceAndQueues();
    _CreateSwapchain(window.GetWidth(), window.GetHeight());
    _CreateImageViews();
    _CreateRenderPass();
    _CreateGraphicsPipeline();
    _CreateFramebuffers();
    _CreateCommandPool();
    _CreateCommandBuffers();
    _CreateSemaphores();
}


void VkBackend::DrawFrame()
{
    const ui32 imageIndex = m_device->acquireNextImageKHR(m_swapchain.get(), std::numeric_limits<ui64>::max(), m_imageAvailableSemaphore.get(), nullptr);
    // NOTE: I guess constexpr is useless because of &dstStageMask
    constexpr vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{ .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &m_imageAvailableSemaphore.get(),
                               .pWaitDstStageMask = &dstStageMask,
                               .commandBufferCount = 1,
                               .pCommandBuffers = &m_commandBuffers[imageIndex],
                               .signalSemaphoreCount = 1,
                               .pSignalSemaphores = &m_renderFinishedSemaphore.get() };

    m_graphicsQueue.submit(submitInfo, nullptr);

    vk::PresentInfoKHR presentInfo{ .waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &m_renderFinishedSemaphore.get(),
                                    .swapchainCount = 1,
                                    .pSwapchains = &m_swapchain.get(),
                                    .pImageIndices = &imageIndex };

    m_presentQueue.presentKHR(presentInfo);
}

void VkBackend::WaitIdle() const
{
    m_device->waitIdle();
}


void VkBackend::_CreateVkInstance(const ui32 apiVersion)
{
    if (kEnableValidationLayers && _checkValidationLayersSupport() == false) {
        throw std::runtime_error("Validation layers requested but not available!");
    }

    _checkAPIVersionSupport(apiVersion);

    const auto extensions = _getRequiredExtensions();

    const vk::ApplicationInfo appInfo{ .pApplicationName = "VkTriangle",
                                       .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
                                       .pEngineName = "No Engine",
                                       .engineVersion = VK_MAKE_VERSION(0, 1, 0),
                                       .apiVersion = apiVersion };

    vk::InstanceCreateInfo instanceInfo{ .pApplicationInfo = &appInfo,
                                         .enabledExtensionCount = static_cast<ui32>(extensions.size()),
                                         .ppEnabledExtensionNames = extensions.data() };

    // NOTE: Reuse create info?
    const auto messengerInfo = _makeDebugUtilsMessengerCreateInfo();
    if (kEnableValidationLayers) {
        instanceInfo.enabledLayerCount = static_cast<ui32>(kValidationLayers.size());
        instanceInfo.ppEnabledLayerNames = kValidationLayers.data();
        instanceInfo.pNext = &messengerInfo;
    } else {
        instanceInfo.enabledLayerCount = 0;
    }

    m_vkInstance = vk::createInstanceUnique(instanceInfo);
}

void VkBackend::_SetupDebugMessenger()
{
    if (kEnableValidationLayers == false) {
        return;
    }

    const auto messengerInfo = _makeDebugUtilsMessengerCreateInfo();
    m_debugMessenger = m_vkInstance->createDebugUtilsMessengerEXTUnique(messengerInfo);
}

// NOTE: Depends on Window class (GLFWindow)
// TODO: Move glfwCreateWindowSurface() to Window class ?
void VkBackend::_CreateSurface(GLFWwindow* windowHandle)
{
    // NOTE: Don't know if there is a way to make it without 'tmp'
    VkSurfaceKHR tmp;

    if (glfwCreateWindowSurface(m_vkInstance.get(), windowHandle, nullptr, &tmp) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create a window surface!");
    }
    m_surface = vk::UniqueSurfaceKHR(tmp, m_vkInstance.get());
}

void VkBackend::_SelectPhysicalDevice()
{
    const auto physicalDevices = m_vkInstance->enumeratePhysicalDevices();
    // NOTE: Do I need to check this? Or vulkan.hpp will do this for me?
    if (physicalDevices.size() == 0) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    for (const auto& device : physicalDevices) {
        if (_isDeviceSuitable(device, m_surface)) {
            m_physicalDevice = device;
            break;
        }
    }

    if (m_physicalDevice == vk::PhysicalDevice()) {
        throw std::runtime_error("Failed to find a suitable GPU!");
    }
}

void VkBackend::_CreateLogicalDeviceAndQueues()
{
    // TODO: We get queue indices when we select physicalDevice. Need to remove this redundant work.
    const auto indices = _getRequiredQueueFamilies(m_physicalDevice, m_surface);

    const std::unordered_set<ui32> uniqueQueueFamilies{ indices.graphicsFamily.value(),
                                                        indices.presentFamily.value() };
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueQueueFamilies.size());

    constexpr float queuePriority = 1.0f; // NOTE: same for every queue
    for (ui32 queueFamily : uniqueQueueFamilies) {
        queueInfos.push_back(vk::DeviceQueueCreateInfo{ .queueFamilyIndex = queueFamily,
                                                        .queueCount = 1,
                                                        .pQueuePriorities = &queuePriority });
    }

    vk::PhysicalDeviceFeatures device_features{}; // NOTE: empty for now

    // DIFFERENCE: Skipped enabling validation layers for device, since there is no need to do that in modern Vulkan
    vk::DeviceCreateInfo deviceinfo{ .queueCreateInfoCount = static_cast<ui32>(queueInfos.size()),
                                     .pQueueCreateInfos = queueInfos.data(),
                                     .enabledExtensionCount = static_cast<ui32>(kDeviceExtensions.size()),
                                     .ppEnabledExtensionNames = kDeviceExtensions.data(),
                                     .pEnabledFeatures = &device_features };

    m_device = m_physicalDevice.createDeviceUnique(deviceinfo);

    // NOTE: m_graphicsQueue and m_presentQueue can hold the same value
    m_graphicsQueue = m_device->getQueue(indices.graphicsFamily.value(), 0);
    m_presentQueue = m_device->getQueue(indices.presentFamily.value(), 0);
}

// TODO: Remove this width/height shit
void VkBackend::_CreateSwapchain(ui32 width, ui32 height)
{
    const auto swapchainSupport = _querySwapchainSupport(m_physicalDevice, m_surface);

    const auto surfaceFormat = _chooseSurfaceFormat(swapchainSupport.formats);
    const auto presentMode = _choosePresentMode(swapchainSupport.presentModes);
    const auto extent = _chooseSurfaceExtent(swapchainSupport.capabilities, width, height);
    // NOTE: Quiestionable
    ui32 imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount != 0 && imageCount > swapchainSupport.capabilities.maxImageCount) {
        imageCount = swapchainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchainInfo{ .surface = m_surface.get(),
                                              .minImageCount = imageCount,
                                              .imageFormat = surfaceFormat.format,
                                              .imageColorSpace = surfaceFormat.colorSpace,
                                              .imageExtent = extent,
                                              .imageArrayLayers = 1, // Always 1, unless it's 3D stereoscopic app
                                              .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                              .preTransform = swapchainSupport.capabilities.currentTransform, // NOTE: Or no transform?
                                              .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                              .presentMode = presentMode,
                                              .clipped = VK_TRUE,
                                              .oldSwapchain = nullptr };

    // TODO: Nice one tutorial, query same shit for third time
    const auto indices = _getRequiredQueueFamilies(m_physicalDevice, m_surface);
    const ui32 familyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        swapchainInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapchainInfo.queueFamilyIndexCount = 2;
        swapchainInfo.pQueueFamilyIndices = familyIndices;
    } else {
        swapchainInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    m_swapchain = m_device->createSwapchainKHRUnique(swapchainInfo);
    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImages = m_device->getSwapchainImagesKHR(m_swapchain.get());
}

void VkBackend::_CreateImageViews()
{
    m_swapchainImageViews.reserve(m_swapchainImages.size());

    vk::ComponentMapping componentMapping{ .r = vk::ComponentSwizzle::eIdentity,
                                           .g = vk::ComponentSwizzle::eIdentity,
                                           .b = vk::ComponentSwizzle::eIdentity,
                                           .a = vk::ComponentSwizzle::eIdentity };
    // NOTE: No mipmap rn. Layers are for stereographic 3D app.
    vk::ImageSubresourceRange subresiurceRange{ .aspectMask = vk::ImageAspectFlagBits::eColor,
                                                .baseMipLevel = 0,
                                                .levelCount = 1,
                                                .baseArrayLayer = 0,
                                                .layerCount = 1 };

    vk::ImageViewCreateInfo imageViewInfo{ .viewType = vk::ImageViewType::e2D,
                                           .format = m_swapchainFormat,
                                           .components = componentMapping,
                                           .subresourceRange = subresiurceRange };

    for (const auto& swapchainImage : m_swapchainImages) {
        imageViewInfo.image = swapchainImage;
        m_swapchainImageViews.push_back(m_device->createImageViewUnique(imageViewInfo));
    }
}

void VkBackend::_CreateRenderPass()
{
    vk::AttachmentDescription colorAttachment{ .format = m_swapchainFormat,
                                               .samples = vk::SampleCountFlagBits::e1,
                                               .loadOp = vk::AttachmentLoadOp::eClear,
                                               .storeOp = vk::AttachmentStoreOp::eStore,
                                               .initialLayout = vk::ImageLayout::eUndefined,
                                               .finalLayout = vk::ImageLayout::ePresentSrcKHR };
    // NOTE: This attachment references fragment shader 'layout(location=0) out vec4 outColor' string
    vk::AttachmentReference colorRef{ .attachment = 0,
                                      .layout = vk::ImageLayout::eColorAttachmentOptimal };
    // NOTE: I'm not quite sure what this shit does
    vk::SubpassDependency dependency{ .srcSubpass = VK_SUBPASS_EXTERNAL,
                                      .dstSubpass = 0,
                                      .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                      /*.srcAccessMask = vk::AccessFlags(),*/ // TODO: I DONT FUCKING KNOW WHAT FLAG CORRESPONDS TO '0' NICE TUTORIAL
                                      .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                      .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite };

    vk::SubpassDescription subpass{ .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &colorRef };

    vk::RenderPassCreateInfo renderPassInfo{ .attachmentCount = 1,
                                             .pAttachments = &colorAttachment,
                                             .subpassCount = 1,
                                             .pSubpasses = &subpass,
                                             .dependencyCount = 1,
                                             .pDependencies = &dependency };

    m_renderPass = m_device->createRenderPassUnique(renderPassInfo);
}

void VkBackend::_CreateGraphicsPipeline()
{
    const auto vertShaderCode = _readShaderFile(kShaderVertexPath);
    const auto fragShaderCode = _readShaderFile(kShaderFragmentPath);

    const auto vertShaderModule = _createShaderModule(vertShaderCode, m_device);
    const auto fragShaderModule = _createShaderModule(fragShaderCode, m_device);

    // NOTE: .pSpecializationInfo allows specify values for shader constants, it can be more efficient
    vk::PipelineShaderStageCreateInfo vertShaderStage{ .stage = vk::ShaderStageFlagBits::eVertex,
                                                       .module = vertShaderModule.get(),
                                                       .pName = "main" };

    vk::PipelineShaderStageCreateInfo fragShaderStage{ .stage = vk::ShaderStageFlagBits::eFragment,
                                                       .module = fragShaderModule.get(),
                                                       .pName = "main" };
    vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStage, fragShaderStage };

    vk::PipelineVertexInputStateCreateInfo vertexInputState{ .vertexBindingDescriptionCount = 0,
                                                             .pVertexBindingDescriptions = nullptr,
                                                             .vertexAttributeDescriptionCount = 0,
                                                             .pVertexAttributeDescriptions = nullptr };

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{ .topology = vk::PrimitiveTopology::eTriangleList,
                                                                 .primitiveRestartEnable = VK_FALSE };
    vk::Viewport viewport{ .x = 0.0f,
                           .y = 0.0f,
                           .width = static_cast<float>(m_swapchainExtent.width),
                           .height = static_cast<float>(m_swapchainExtent.height),
                           .minDepth = 0.0f,
                           .maxDepth = 1.0f };

    vk::Rect2D scissor{ .offset = {0, 0},
                        .extent = m_swapchainExtent };

    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1,
                                                       .pViewports = &viewport,
                                                       .scissorCount = 1,
                                                       .pScissors = &scissor };

    vk::PipelineRasterizationStateCreateInfo rasterizationState{ .depthClampEnable = VK_FALSE,
                                                                 .rasterizerDiscardEnable = VK_FALSE,
                                                                 .polygonMode = vk::PolygonMode::eFill,
                                                                 .cullMode = vk::CullModeFlagBits::eBack,
                                                                 .frontFace = vk::FrontFace::eClockwise,
                                                                 .depthBiasEnable = VK_FALSE,
                                                                 .depthBiasConstantFactor = 0.0f,
                                                                 .depthBiasClamp = 0.0f,
                                                                 .depthBiasSlopeFactor = 0.0f,
                                                                 .lineWidth = 1.0f };

    vk::PipelineMultisampleStateCreateInfo multisampleState{ .rasterizationSamples = vk::SampleCountFlagBits::e1,
                                                             .sampleShadingEnable = VK_FALSE };

    //vk::PipelineDepthStencilStateCreateInfo

    vk::ColorComponentFlags colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
        | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendAttachmentState colorBlendAttachment{ .blendEnable = VK_FALSE,
                                                                .colorWriteMask = colorWriteMask };

    vk::PipelineColorBlendStateCreateInfo colorBlendState{ .logicOpEnable = VK_FALSE,
                                                           .logicOp = vk::LogicOp::eCopy,
                                                           .attachmentCount = 1,
                                                           .pAttachments = &colorBlendAttachment };
    // NOTE: No uniform variables in shaders for now
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};

    m_pipelineLayout = m_device->createPipelineLayoutUnique(pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo graphicsPipelineInfo{ .stageCount = 2,
                                                         .pStages = shaderStages,
                                                         .pVertexInputState = &vertexInputState,
                                                         .pInputAssemblyState = &inputAssemblyState,
                                                         .pViewportState = &viewportState,
                                                         .pRasterizationState = &rasterizationState,
                                                         .pMultisampleState = &multisampleState,
                                                         .pDepthStencilState = nullptr,
                                                         .pColorBlendState = &colorBlendState,
                                                         .pDynamicState = nullptr,
                                                         .layout = m_pipelineLayout.get(),
                                                         .renderPass = m_renderPass.get(),
                                                         .subpass = 0 };

    m_pipeline = m_device->createGraphicsPipelineUnique(nullptr, graphicsPipelineInfo);
}

void VkBackend::_CreateFramebuffers()
{
    m_framebuffers.reserve(m_swapchainImageViews.size());

    vk::ImageView attachments[1];

    vk::FramebufferCreateInfo framebufferInfo{ .renderPass = m_renderPass.get(),
                                               .attachmentCount = 1,
                                               .pAttachments = attachments,
                                               .width = m_swapchainExtent.width,
                                               .height = m_swapchainExtent.height,
                                               .layers = 1 };

    for (const auto& imageView : m_swapchainImageViews) {
        attachments[0] = imageView.get();
        m_framebuffers.push_back(m_device->createFramebufferUnique(framebufferInfo));
    }
}

void VkBackend::_CreateCommandPool()
{
    // NOTE: Query same shit for the 4th time
    const auto queueFamilyIndices = _getRequiredQueueFamilies(m_physicalDevice, m_surface);

    vk::CommandPoolCreateInfo commandPoolInfo{ .flags = vk::CommandPoolCreateFlags(),
                                               .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value() };

    m_commandPool = m_device->createCommandPoolUnique(commandPoolInfo);
}

void VkBackend::_CreateCommandBuffers()
{
    vk::CommandBufferAllocateInfo commandBufferInfo{ .commandPool = m_commandPool.get(),
                                                     .level = vk::CommandBufferLevel::ePrimary,
                                                     .commandBufferCount = static_cast<ui32>(m_framebuffers.size()) };

    m_commandBuffers = m_device->allocateCommandBuffers(commandBufferInfo);

    vk::Rect2D renderArea{ .offset = {0, 0},
                           .extent = m_swapchainExtent };

    vk::ClearValue clearColor = vk::ClearColorValue(std::array<float, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });

    // NOTE: This should be moved to its own method (StartFrame() ?)
    for (int i = 0; i < m_commandBuffers.size(); ++i) {
        const auto& commandBuffer = m_commandBuffers[i];

        vk::CommandBufferBeginInfo beginInfo{};

        vk::RenderPassBeginInfo renderPassInfo{ .renderPass = m_renderPass.get(),
                                                .framebuffer = m_framebuffers[i].get(),
                                                .renderArea = renderArea,
                                                .clearValueCount = 1,
                                                .pClearValues = &clearColor };

        commandBuffer.begin(beginInfo);
        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline.get());
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }
}

void VkBackend::_CreateSemaphores()
{
    vk::SemaphoreCreateInfo semaphoreInfo{};

    m_imageAvailableSemaphore = m_device->createSemaphoreUnique(semaphoreInfo);
    m_renderFinishedSemaphore = m_device->createSemaphoreUnique(semaphoreInfo);
}

}



void _checkAPIVersionSupport(const ui32 requestedVersion)
{
    auto vkVersionToString = [](const ui32 ver)
    {
        auto major = VK_VERSION_MAJOR(ver);
        auto minor = VK_VERSION_MINOR(ver);
        auto patch = VK_VERSION_PATCH(ver);
        return std::to_string(major) + '.' + std::to_string(minor) + '.' + std::to_string(patch);
    };

    const ui32 currentVersion{ vk::enumerateInstanceVersion() };

    if (currentVersion < requestedVersion) {
        std::cout << "Requested Vulkan API version v" << vkVersionToString(requestedVersion) <<
            " but the Vulkan implementation on this device only supports v" << vkVersionToString(currentVersion) << '\n';
    } else if (currentVersion > requestedVersion) {
        std::cout << "Requested Vulkan API version v" << vkVersionToString(requestedVersion) <<
            ", but the Vulkan implementation on this device can actually support newer version v" <<
            vkVersionToString(currentVersion) << '\n';
    }
}

// NOTE: Depends on Window class (GLFWindow)
std::vector<const char*> _getRequiredExtensions()
{
    ui32 glfwExtensionCount = 0;
    const auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (kEnableValidationLayers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool _checkValidationLayersSupport()
{
    const auto availableLayers = vk::enumerateInstanceLayerProperties();

    return std::all_of(kValidationLayers.begin(), kValidationLayers.end(),
                       [&availableLayers](const char* required) {
                           return std::find_if(availableLayers.begin(), availableLayers.end(),
                                               [&required](const vk::LayerProperties& available) {
                                                   return std::strcmp(required, available.layerName) == 0;
                                               }) != availableLayers.end();
                       });
}

vk::DebugUtilsMessengerCreateInfoEXT _makeDebugUtilsMessengerCreateInfo()
{
    vk::DebugUtilsMessageSeverityFlagsEXT severity{ vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
                                                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                                                    | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };
    vk::DebugUtilsMessageTypeFlagsEXT type{ vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                                            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                                            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance };
    vk::DebugUtilsMessengerCreateInfoEXT messengerInfo{ .messageSeverity = severity,
                                                        .messageType = type,
                                                        .pfnUserCallback = _debugMessageCallback };
    return messengerInfo;
}


// NOTE: Fuckin surface
bool _isDeviceSuitable(const vk::PhysicalDevice device, const vk::UniqueSurfaceKHR& surface)
{
    bool isQueueFamiliesSupported = _getRequiredQueueFamilies(device, surface).isComplete();
    bool isExtensionsSupported = _checkPhysicalDeviceExtensionSupport(device);
    bool isSwapChainAdequate = false;
    if (isExtensionsSupported) {
        auto swapChainSupport = _querySwapchainSupport(device, surface);
        // NOTE: Move this to a SwapChainSupportDetails method? Similar to isComplete()
        isSwapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return isQueueFamiliesSupported && isExtensionsSupported && isSwapChainAdequate;
}

// NOTE: Depends on m_surface, make it private method ?
QueueFamilyIndices _getRequiredQueueFamilies(const vk::PhysicalDevice device, const vk::UniqueSurfaceKHR& surface)
{
    QueueFamilyIndices indices;

    // NOTE: This can replace indices for the queues already found, may be add smth like:
    //   if (indices.graphics_family.has_value() == false && queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
    for (ui32 i = 0; const auto & queueFamily : device.getQueueFamilyProperties()) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        if (device.getSurfaceSupportKHR(i, surface.get())) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }
        ++i;
    }

    return indices;
}

bool _checkPhysicalDeviceExtensionSupport(const vk::PhysicalDevice device)
{
    const auto availableExtensions = device.enumerateDeviceExtensionProperties();

    return std::all_of(kDeviceExtensions.begin(), kDeviceExtensions.end(),
                       [&availableExtensions](const char* required) {
                           return std::find_if(availableExtensions.begin(), availableExtensions.end(),
                                               [&required](const vk::ExtensionProperties& available) {
                                                   return std::strcmp(required, available.extensionName) == 0;
                                               }) != availableExtensions.end();
                       });
}


SwapchainSupportDetails _querySwapchainSupport(const vk::PhysicalDevice device, const vk::UniqueSurfaceKHR& surface)
{
    return { .formats = device.getSurfaceFormatsKHR(surface.get()),
             .presentModes = device.getSurfacePresentModesKHR(surface.get()),
             .capabilities = device.getSurfaceCapabilitiesKHR(surface.get()) };
}

vk::SurfaceFormatKHR _chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
{
    for (const auto& format : availableFormats) {
        if (format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return format;
        }
    }

    if(availableFormats.front().format == vk::Format::eUndefined)
        return { vk::Format::eB8G8R8A8Srgb, vk::ColorSpaceKHR::eSrgbNonlinear };

    return availableFormats.front();
}

vk::PresentModeKHR _choosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
{
    // TODO: Rewrite with ranges?
    auto mode = std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eMailbox);
    if (mode != availablePresentModes.end())
        return *mode;

    return vk::PresentModeKHR::eFifo;
    //return vk::PresentModeKHR::eImmediate;
}

// NOTE: This 'width', 'height' shit looks ugly
vk::Extent2D _chooseSurfaceExtent(const vk::SurfaceCapabilitiesKHR& capabilities, ui32 width, ui32 height)
{
    if (capabilities.currentExtent.width != std::numeric_limits<ui32>::max()) {
        return capabilities.currentExtent;
    }

    return { .width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
             .height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height) };
}


std::vector<char> _readShaderFile(const std::string_view shaderPath)
{
    auto size = std::filesystem::file_size(shaderPath);
    std::vector<char> buffer(size);

    std::ifstream shaderFile(shaderPath, std::ios::binary | std::ios::in);
    shaderFile.read(buffer.data(), size);

    return buffer;
}

vk::UniqueShaderModule _createShaderModule(const std::vector<char>& shaderCode, const vk::UniqueDevice& device)
{
    // NOTE: May be read shader file as 'ui32' instead of 'char'
    vk::ShaderModuleCreateInfo shaderModuleInfo{ .codeSize = shaderCode.size(),
                                                 .pCode = reinterpret_cast<const ui32*>(shaderCode.data()) };

    return device->createShaderModuleUnique(shaderModuleInfo);
}
