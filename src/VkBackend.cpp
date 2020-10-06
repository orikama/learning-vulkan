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
#include <chrono>

#define GLM_FORCE_RADIANS
//#define GLM_FORCE_LEFT_HANDED
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>


// TODO: Remove globals
constexpr i32 kMaxFramesInFlight = 2;
constexpr i64 kSyncObjectTimeout = std::numeric_limits<ui64>::max();

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
// TODO: Separate transfer queue
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


struct Vertex
{
    glm::vec2 position;
    glm::vec3 color;

    static constexpr vk::VertexInputBindingDescription GetBindingDescription() noexcept
    {
        vk::VertexInputBindingDescription bindingDescription{ .binding = kBinding,
                                                              .stride = sizeof(Vertex),
                                                              .inputRate = vk::VertexInputRate::eVertex };
        //static_assert(sizeof(Vertex) == 192);
        //std::cout << sizeof(Vertex) << '\n';

        return bindingDescription;
    }

    static constexpr std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescription() noexcept
    {
        vk::VertexInputAttributeDescription positionAttribute{ .location = 0,
                                                               .binding = kBinding,
                                                               .format = vk::Format::eR32G32Sfloat,
                                                               .offset = offsetof(Vertex, position) };
        vk::VertexInputAttributeDescription colorAttribute{ .location = 1,
                                                            .binding = kBinding,
                                                            .format = vk::Format::eR32G32B32Sfloat,
                                                            .offset = offsetof(Vertex, color) };
        return { positionAttribute, colorAttribute };
    }

private:
    static const ui32 kBinding = 0; // FINDOUT: WTF is this
};

struct UBO_MVP
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 projection;
};


const std::vector<Vertex> kTriangleVertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<ui16> kTriangleIndices = {
    0, 1, 2, 2, 3, 0
};


auto _checkAPIVersionSupport(const ui32 requestedVersion)   -> void;
auto _getRequiredExtensions()                               -> std::vector<const char*>;
auto _checkValidationLayersSupport()                        -> bool;
auto _makeDebugUtilsMessengerCreateInfo()                   -> vk::DebugUtilsMessengerCreateInfoEXT;

auto _isDeviceSuitable(const vk::PhysicalDevice& device,
                       const vk::SurfaceKHR& surface)                        -> bool;
auto _getRequiredQueueFamilies(const vk::PhysicalDevice& device,
                               const vk::SurfaceKHR& surface)                -> QueueFamilyIndices;
auto _checkPhysicalDeviceExtensionSupport(const vk::PhysicalDevice& device)  -> bool;

auto _querySwapchainSupport(const vk::PhysicalDevice& device,
                            const vk::SurfaceKHR& surface)              -> SwapchainSupportDetails;
auto _chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)    -> vk::SurfaceFormatKHR;
auto _choosePresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)   -> vk::PresentModeKHR;
auto _chooseSurfaceExtent(const vk::SurfaceCapabilitiesKHR& capabilities, ui32 width, ui32 height) -> vk::Extent2D;

auto _readShaderFile(const std::string_view shaderPath)         -> std::vector<char>;
auto _createShaderModule(const std::vector<char>& shaderCode,
                         const vk::Device& device)              -> vk::UniqueShaderModule;

auto _findMemoryTypeIndex(const vk::PhysicalDevice& physicalDevice,
                          const ui32 memoryTypeBits,
                          const vk::MemoryPropertyFlags properties)                             -> ui32;
// NOTE: I think i should make it private method or a friend function
auto _createBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device,
                   const vk::DeviceSize bufferSize,
                   const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties,
                   vk::Buffer& buffer, vk::DeviceMemory& bufferMemory)                          -> void;
auto _copyBuffer(const vk::CommandPool& commandPool, const vk::Device& device, const vk::Queue& queue,
                 vk::Buffer& source, vk::Buffer& destination, vk::DeviceSize size)              -> void;



namespace vulkan
{

void VkBackend::Init(const Window& window)
{
    m_frameCounter = 0;
    m_currentFrameData = 0;

    _CreateInstance(/*VK_API_VERSION_1_2*/VK_MAKE_VERSION(1, 2, 135));
    _SetupDebugMessenger();
    _CreateSurface(window.GetWindowHandle());
    _SelectPhysicalDevice();
    _CreateLogicalDeviceAndQueues();
    _CreateSwapchain(window.GetWidth(), window.GetHeight());
    _CreateImageViews();
    _CreateRenderPass();

    _CreateDescriptorSetLayout();
    _CreateGraphicsPipeline();

    _CreateFramebuffers();
    _CreateCommandPool();

    _CreateVertexBuffer();
    _CreateIndexBuffer();
    _CreateUniformBuffers();

    _CreateDescriptorPool();
    _CreateDescriptorSets();

    _CreateCommandBuffers();
    _CreateSyncPrimitives();
}

void VkBackend::Shutdown()
{
    for (int i = 0; i < kMaxFramesInFlight; ++i) {
        m_device.destroySemaphore(m_imageAvailableSemaphores[i]);
        m_device.destroySemaphore(m_renderFinishedSemaphores[i]);
        m_device.destroyFence(m_inFlightFences[i]);
    }

    m_device.destroyBuffer(m_indexBuffer);
    m_device.freeMemory(m_indexBufferMemory);
    m_device.destroyBuffer(m_vertexBuffer);
    m_device.freeMemory(m_vertexBufferMemory);

    _CleanupSwapchain();

    m_device.destroyDescriptorSetLayout(m_descriptorSetLayout);

    m_device.destroyCommandPool(m_commandPool);
    m_device.destroy();

    if (kEnableValidationLayers) {
        m_instance.destroyDebugUtilsMessengerEXT(m_debugMessenger);
    }

    m_instance.destroySurfaceKHR(m_surface);
    m_instance.destroy();
}


void VkBackend::DrawFrame()
{
    m_device.waitForFences(1, &m_inFlightFences[m_currentFrameData], VK_TRUE, kSyncObjectTimeout);
    m_device.resetFences(1, &m_inFlightFences[m_currentFrameData]);

    const ui32 imageIndex = m_device.acquireNextImageKHR(m_swapchain, kSyncObjectTimeout,
                                                         m_imageAvailableSemaphores[m_currentFrameData], nullptr);

    _UpdateUniformBuffers(imageIndex);

    // NOTE: I guess constexpr is useless because of &dstStageMask
    constexpr vk::PipelineStageFlags dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{ .waitSemaphoreCount = 1,
                               .pWaitSemaphores = &m_imageAvailableSemaphores[m_currentFrameData],
                               .pWaitDstStageMask = &dstStageMask,
                               .commandBufferCount = 1,
                               .pCommandBuffers = &m_commandBuffers[imageIndex],
                               .signalSemaphoreCount = 1,
                               .pSignalSemaphores = &m_renderFinishedSemaphores[m_currentFrameData] };

    m_graphicsQueue.submit(submitInfo, m_inFlightFences[m_currentFrameData]);

    vk::PresentInfoKHR presentInfo{ .waitSemaphoreCount = 1,
                                    .pWaitSemaphores = &m_renderFinishedSemaphores[m_currentFrameData],
                                    .swapchainCount = 1,
                                    .pSwapchains = &m_swapchain,
                                    .pImageIndices = &imageIndex };

    m_presentQueue.presentKHR(presentInfo);

    ++m_frameCounter;
    m_currentFrameData = m_frameCounter % kMaxFramesInFlight;
    //std::cout << m_frameCounter << ' ' << m_currentFrameData << '\n';
}

void VkBackend::WaitIdle() const
{
    m_device.waitIdle();
}


void VkBackend::_CreateInstance(const ui32 apiVersion)
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

    m_instance = vk::createInstance(instanceInfo);
}

void VkBackend::_SetupDebugMessenger()
{
    if (kEnableValidationLayers == false) {
        return;
    }

    const auto messengerInfo = _makeDebugUtilsMessengerCreateInfo();
    m_debugMessenger = m_instance.createDebugUtilsMessengerEXT(messengerInfo);
}

// NOTE: Depends on Window class (GLFWindow)
// TODO: Move glfwCreateWindowSurface() to Window class ?
void VkBackend::_CreateSurface(GLFWwindow* windowHandle)
{
    // NOTE: Don't know if there is a way to make it without 'tmp'
    VkSurfaceKHR tmp;

    if (glfwCreateWindowSurface(m_instance, windowHandle, nullptr, &tmp) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create a window surface!");
    }

    m_surface = tmp;
}

void VkBackend::_SelectPhysicalDevice()
{
    const auto physicalDevices = m_instance.enumeratePhysicalDevices();
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

    constexpr f32 queuePriority = 1.0f; // NOTE: same for every queue
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

    m_device = m_physicalDevice.createDevice(deviceinfo);

    // NOTE: m_graphicsQueue and m_presentQueue can hold the same value
    m_graphicsQueue = m_device.getQueue(indices.graphicsFamily.value(), 0);
    m_presentQueue = m_device.getQueue(indices.presentFamily.value(), 0);
}

// TODO: Remove this width/height shit
void VkBackend::_CreateSwapchain(ui32 width, ui32 height)
{
    const auto swapchainSupport = _querySwapchainSupport(m_physicalDevice, m_surface);

    const auto surfaceFormat = _chooseSurfaceFormat(swapchainSupport.formats);
    const auto presentMode = _choosePresentMode(swapchainSupport.presentModes);
    const auto extent = _chooseSurfaceExtent(swapchainSupport.capabilities, width, height);
    // NOTE: Quiestionable
    // TODO: Shouldn't imageCount be in sync with kMaxFramesInFlight ?
    ui32 imageCount = swapchainSupport.capabilities.minImageCount + 1;
    if (swapchainSupport.capabilities.maxImageCount != 0 && imageCount > swapchainSupport.capabilities.maxImageCount) {
        imageCount = swapchainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR swapchainInfo{ .surface = m_surface,
                                              .minImageCount = imageCount,
                                              .imageFormat = surfaceFormat.format,
                                              .imageColorSpace = surfaceFormat.colorSpace,
                                              .imageExtent = extent,
                                              .imageArrayLayers = 1, // NOTE: Always 1, unless it's 3D stereoscopic app
                                              .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
                                              .preTransform = swapchainSupport.capabilities.currentTransform, // NOTE: Seems like this is for mobile
                                              .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque, // NOTE: 'Opaque' is not guaranteed to be supported
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

    m_swapchain = m_device.createSwapchainKHR(swapchainInfo);
    m_swapchainFormat = surfaceFormat.format;
    m_swapchainExtent = extent;

    m_swapchainImages = m_device.getSwapchainImagesKHR(m_swapchain);
}

void VkBackend::_CreateImageViews()
{
    vk::ComponentMapping componentMapping{ .r = vk::ComponentSwizzle::eIdentity,
                                           .g = vk::ComponentSwizzle::eIdentity,
                                           .b = vk::ComponentSwizzle::eIdentity,
                                           .a = vk::ComponentSwizzle::eIdentity };
    // NOTE: No mipmap rn. Layers are for stereographic 3D app.
    vk::ImageSubresourceRange subresourceRange{ .aspectMask = vk::ImageAspectFlagBits::eColor,
                                                .baseMipLevel = 0,
                                                .levelCount = 1,
                                                .baseArrayLayer = 0,
                                                .layerCount = 1 };

    vk::ImageViewCreateInfo imageViewInfo{ .viewType = vk::ImageViewType::e2D,
                                           .format = m_swapchainFormat,
                                           .components = componentMapping,
                                           .subresourceRange = subresourceRange };

    m_swapchainImageViews.reserve(m_swapchainImages.size());

    for (const auto& swapchainImage : m_swapchainImages) {
        imageViewInfo.image = swapchainImage;
        m_swapchainImageViews.push_back(m_device.createImageView(imageViewInfo));
    }
}

void VkBackend::_CreateRenderPass()
{
    vk::AttachmentDescription colorAttachment{ .format = m_swapchainFormat,
                                               .samples = vk::SampleCountFlagBits::e1,
                                               .loadOp = vk::AttachmentLoadOp::eClear,
                                               .storeOp = vk::AttachmentStoreOp::eStore,
                                               .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
                                               .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
                                               .initialLayout = vk::ImageLayout::eUndefined,    // NOTE: undefined ???
                                               .finalLayout = vk::ImageLayout::ePresentSrcKHR };
    // NOTE: This attachment references fragment shader 'layout(location=0) out vec4 outColor' string
    vk::AttachmentReference colorRef{ .attachment = 0,
                                      .layout = vk::ImageLayout::eColorAttachmentOptimal };

    // NOTE: I'm not quite sure what this shit does
    //vk::SubpassDependency dependency{ .srcSubpass = VK_SUBPASS_EXTERNAL,
    //                                  .dstSubpass = 0,
    //                                  .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
    //                                  .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput,
    //                                  /*.srcAccessMask = vk::AccessFlags(),*/ // TODO: I DONT FUCKING KNOW WHAT FLAG CORRESPONDS TO '0' NICE TUTORIAL
    //                                  .srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead,
    //                                  .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite };

    vk::SubpassDescription subpass{ .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
                                    .colorAttachmentCount = 1,
                                    .pColorAttachments = &colorRef };

    vk::RenderPassCreateInfo renderPassInfo{ .attachmentCount = 1,
                                             .pAttachments = &colorAttachment,
                                             .subpassCount = 1,
                                             .pSubpasses = &subpass,
                                             /*.dependencyCount = 1,
                                             .pDependencies = &dependency*/ };

    m_renderPass = m_device.createRenderPass(renderPassInfo);
}


void VkBackend::_CreateDescriptorSetLayout()
{
    // NOTE: This is where the shader reflection can help
    vk::DescriptorSetLayoutBinding uboLayoutBinding{ .binding = 0,
                                                     .descriptorType = vk::DescriptorType::eUniformBuffer,
                                                     .descriptorCount = 1,
                                                     .stageFlags = vk::ShaderStageFlagBits::eVertex,
                                                     .pImmutableSamplers = nullptr};

    vk::DescriptorSetLayoutCreateInfo descriptorLayoutInfo{ .bindingCount = 1,
                                                            .pBindings = &uboLayoutBinding };

    m_descriptorSetLayout = m_device.createDescriptorSetLayout(descriptorLayoutInfo);
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

    constexpr auto bindingDescription = Vertex::GetBindingDescription();
    constexpr auto attributeDescription = Vertex::GetAttributeDescription();

    vk::PipelineVertexInputStateCreateInfo vertexInputState{ .vertexBindingDescriptionCount = 1,
                                                             .pVertexBindingDescriptions = &bindingDescription,
                                                             .vertexAttributeDescriptionCount = static_cast<ui32>(attributeDescription.size()),
                                                             .pVertexAttributeDescriptions = attributeDescription.data() };

    vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState{ .topology = vk::PrimitiveTopology::eTriangleList,
                                                                 .primitiveRestartEnable = VK_FALSE };
    vk::Viewport viewport{ .x = 0.0f,
                           .y = 0.0f,
                           .width = static_cast<f32>(m_swapchainExtent.width),
                           .height = static_cast<f32>(m_swapchainExtent.height),
                           .minDepth = 0.0f,
                           .maxDepth = 1.0f };

    vk::Rect2D scissor{ .offset = {0, 0},
                        .extent = m_swapchainExtent };

    vk::PipelineViewportStateCreateInfo viewportState{ .viewportCount = 1,
                                                       .pViewports = &viewport,
                                                       .scissorCount = 1,
                                                       .pScissors = &scissor };
    // NOTE: How the fuck the inversion of Y-axis affects frontFace (or it can be fixed by changing cullMode to eFront)
    vk::PipelineRasterizationStateCreateInfo rasterizationState{ .depthClampEnable = VK_FALSE,
                                                                 .rasterizerDiscardEnable = VK_FALSE,
                                                                 .polygonMode = vk::PolygonMode::eFill,
                                                                 .cullMode = vk::CullModeFlagBits::eBack,
                                                                 .frontFace = vk::FrontFace::eCounterClockwise, // NOTE: was clockwise until UBO kicked in
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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ .setLayoutCount = 1,
                                                     .pSetLayouts = &m_descriptorSetLayout };

    m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutInfo);

    /*vk::DynamicState dynamicStates[] = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };

    vk::PipelineDynamicStateCreateInfo dynamicStateInfo{ .dynamicStateCount = sizeof(dynamicStates) / sizeof(dynamicStates[0]),
                                                         .pDynamicStates = dynamicStates };*/

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
                                                         .layout = m_pipelineLayout,
                                                         .renderPass = m_renderPass,
                                                         .subpass = 0 };
    // NOTE: Idk why I need this cast only there, everywhere else it just works LOOOOOOOOOOOOOOOOOOOOOOOOOOOL
    m_pipeline = (vk::Pipeline&&)m_device.createGraphicsPipeline(nullptr, graphicsPipelineInfo);
}


void VkBackend::_CreateFramebuffers()
{
    m_framebuffers.reserve(m_swapchainImageViews.size());

    vk::ImageView attachments[1];

    vk::FramebufferCreateInfo framebufferInfo{ .renderPass = m_renderPass,
                                               .attachmentCount = 1,
                                               .pAttachments = attachments,
                                               .width = m_swapchainExtent.width,
                                               .height = m_swapchainExtent.height,
                                               .layers = 1 };

    for (const auto& imageView : m_swapchainImageViews) {
        attachments[0] = imageView;
        m_framebuffers.push_back(m_device.createFramebuffer(framebufferInfo));
    }
}

void VkBackend::_CreateCommandPool()
{
    // NOTE: Query same shit for the 4th time
    const auto queueFamilyIndices = _getRequiredQueueFamilies(m_physicalDevice, m_surface);

    vk::CommandPoolCreateInfo commandPoolInfo{ .flags = vk::CommandPoolCreateFlags(),
                                               .queueFamilyIndex = queueFamilyIndices.graphicsFamily.value() };

    m_commandPool = m_device.createCommandPool(commandPoolInfo);
}


void VkBackend::_CreateVertexBuffer()
{
    vk::DeviceSize bufferSize = sizeof(kTriangleVertices[0]) * kTriangleVertices.size();

    constexpr auto stagingProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    _createBuffer(m_physicalDevice, m_device, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, stagingProperties, stagingBuffer, stagingBufferMemory);

    ui8* data = static_cast<ui8*>(m_device.mapMemory(stagingBufferMemory, 0, bufferSize));
    std::memcpy(data, kTriangleVertices.data(), bufferSize);
    //std::copy(kTriangleVertices.data(), kTriangleVertices.data() + bufferSize, data);
    m_device.unmapMemory(stagingBufferMemory);

    constexpr auto bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer;
    _createBuffer(m_physicalDevice, m_device, bufferSize, bufferUsage, vk::MemoryPropertyFlagBits::eDeviceLocal, m_vertexBuffer, m_vertexBufferMemory);

    _copyBuffer(m_commandPool, m_device, m_graphicsQueue, stagingBuffer, m_vertexBuffer, bufferSize);

    m_device.destroyBuffer(stagingBuffer);
    m_device.freeMemory(stagingBufferMemory);
}

// TODO: Why do we need an almost identical to _CreateVertexBuffer() function?
//  Although it seems like I need to use templates for that and put this code to header so fuck that for now
void VkBackend::_CreateIndexBuffer()
{
    const vk::DeviceSize bufferSize = sizeof(kTriangleIndices[0]) * kTriangleIndices.size();

    constexpr auto memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    _createBuffer(m_physicalDevice, m_device, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, memoryProperties, stagingBuffer, stagingBufferMemory);

    ui8* data = static_cast<ui8*>(m_device.mapMemory(stagingBufferMemory, 0, bufferSize));
    std::memcpy(data, kTriangleIndices.data(), bufferSize);
    //std::copy(kTriangleVertices.data(), kTriangleVertices.data() + bufferSize, data);
    m_device.unmapMemory(stagingBufferMemory);

    constexpr auto bufferUsage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer;
    _createBuffer(m_physicalDevice, m_device, bufferSize, bufferUsage, vk::MemoryPropertyFlagBits::eDeviceLocal, m_indexBuffer, m_indexBufferMemory);

    _copyBuffer(m_commandPool, m_device, m_graphicsQueue, stagingBuffer, m_indexBuffer, bufferSize);

    m_device.destroyBuffer(stagingBuffer);
    m_device.freeMemory(stagingBufferMemory);
}

void VkBackend::_CreateUniformBuffers()
{
    const vk::DeviceSize bufferSize = sizeof(UBO_MVP);
    constexpr auto memoryProperties = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    const auto size = m_swapchainImages.size();
    m_uniformBuffers.resize(size);
    m_uniformBuffersMemory.resize(size);

    for (size_t i = 0; i < size; ++i) {
        _createBuffer(m_physicalDevice, m_device, bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, memoryProperties,
                      m_uniformBuffers[i], m_uniformBuffersMemory[i]);
    }
}


void VkBackend::_CreateDescriptorPool()
{
    const auto descriptorCount = static_cast<ui32>(m_swapchainImages.size());

    vk::DescriptorPoolSize poolSize{ .type = vk::DescriptorType::eUniformBuffer,
                                     .descriptorCount = descriptorCount };

    vk::DescriptorPoolCreateInfo poolInfo{ //.flags = vk::DescriptorPoolCreateFlagBits,
                                           .maxSets = descriptorCount,
                                           .poolSizeCount = 1,
                                           .pPoolSizes = &poolSize };

    m_descriptorPool = m_device.createDescriptorPool(poolInfo);
}

void VkBackend::_CreateDescriptorSets()
{
    const auto descriptorCount = static_cast<ui32>(m_swapchainImages.size());

    std::vector<vk::DescriptorSetLayout> layouts(descriptorCount, m_descriptorSetLayout);

    vk::DescriptorSetAllocateInfo descriptorSetInfo{ .descriptorPool = m_descriptorPool,
                                                     .descriptorSetCount = descriptorCount,
                                                     .pSetLayouts = layouts.data() };

    m_descriptorSets = m_device.allocateDescriptorSets(descriptorSetInfo);

    vk::DescriptorBufferInfo descriptorBuffer{ .offset = 0,
                                               .range = sizeof(UBO_MVP) };

    vk::WriteDescriptorSet descriptorWrite{ .dstBinding = 0,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = vk::DescriptorType::eUniformBuffer,
                                            .pBufferInfo = &descriptorBuffer };

    for (ui32 i = 0; i < descriptorCount; ++i) {
        descriptorBuffer.buffer = m_uniformBuffers[i];
        descriptorWrite.dstSet = m_descriptorSets[i];
        m_device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
    }
}


void VkBackend::_CreateCommandBuffers()
{
    vk::CommandBufferAllocateInfo commandBufferInfo{ .commandPool = m_commandPool,
                                                     .level = vk::CommandBufferLevel::ePrimary,
                                                     .commandBufferCount = static_cast<ui32>(m_framebuffers.size()) };

    m_commandBuffers = m_device.allocateCommandBuffers(commandBufferInfo);

    vk::Rect2D renderArea{ .offset = {0, 0},
                           .extent = m_swapchainExtent };

    vk::ClearValue clearColor = vk::ClearColorValue(std::array<f32, 4>{ 0.0f, 0.0f, 0.0f, 0.0f });

    // NOTE: This should be moved to its own method (StartFrame() ?)
    for (int i = 0; i < m_commandBuffers.size(); ++i) {
        const auto& commandBuffer = m_commandBuffers[i];

        vk::CommandBufferBeginInfo beginInfo{};

        vk::RenderPassBeginInfo renderPassInfo{ .renderPass = m_renderPass,
                                                .framebuffer = m_framebuffers[i],
                                                .renderArea = renderArea,
                                                .clearValueCount = 1,
                                                .pClearValues = &clearColor };

        commandBuffer.begin(beginInfo);
        commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        {
            commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_pipeline);

            commandBuffer.bindVertexBuffers(0, m_vertexBuffer, { 0 });
            commandBuffer.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint16);
            commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, 1, &m_descriptorSets[i], 0, nullptr);

            commandBuffer.drawIndexed(static_cast<ui32>(kTriangleIndices.size()), 1, 0, 0, 0);
        }
        commandBuffer.endRenderPass();
        commandBuffer.end();
    }
}

void VkBackend::_CreateSyncPrimitives()
{
    m_imageAvailableSemaphores.reserve(kMaxFramesInFlight);
    m_renderFinishedSemaphores.reserve(kMaxFramesInFlight);
    m_inFlightFences.reserve(kMaxFramesInFlight);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    // NOTE: It's strange that without specifying 'eSignaled' flag it still worked
    vk::FenceCreateInfo fenceInfo{ .flags = vk::FenceCreateFlagBits::eSignaled };

    for (i32 i = 0; i < kMaxFramesInFlight; ++i) {
        m_imageAvailableSemaphores.push_back(m_device.createSemaphore(semaphoreInfo));
        m_renderFinishedSemaphores.push_back(m_device.createSemaphore(semaphoreInfo));
        m_inFlightFences.push_back(m_device.createFence(fenceInfo));
    }
}


void VkBackend::_CleanupSwapchain()
{
    m_device.destroyDescriptorPool(m_descriptorPool);

    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
        m_device.destroyBuffer(m_uniformBuffers[i]);
        m_device.freeMemory(m_uniformBuffersMemory[i]);
    }

    for (auto framebuffer : m_framebuffers) {
        m_device.destroyFramebuffer(framebuffer);
    }

    m_device.freeCommandBuffers(m_commandPool, static_cast<ui32>(m_commandBuffers.size()), m_commandBuffers.data());

    m_device.destroyPipeline(m_pipeline);
    m_device.destroyPipelineLayout(m_pipelineLayout);
    m_device.destroyRenderPass(m_renderPass);

    for (auto imageView : m_swapchainImageViews) {
        m_device.destroyImageView(imageView);
    }

    m_device.destroySwapchainKHR(m_swapchain);
}

//void VkBackend::_RecreateSwapchain()
//{
//    m_device.waitIdle();
//
//    _CleanupSwapchain();
//
//    _CreateSwapchain(1, 1);
//    _CreateImageViews();
//    _CreateRenderPass();
//    // NOTE: Possible to avoid recreation of pipeline, by using dynamic state for viewports and scissor rectnagles
//    // NOTE: May be there are more stuff that can be avoided
//    _CreateGraphicsPipeline();
//    _CreateFramebuffers();
//    _CreateCommandBuffers();
//    _CreateUniformBuffers();
//    _CreateDescriptorPool();
//    _CreateDescriptorSets();
//}


// NOTE: There are more efficient ways to pass data to shaders, like "push constants"
// NOTE: Only model matrix changes every frame, and projection changes when window is resized
void VkBackend::_UpdateUniformBuffers(ui32 imageIndex)
{
    static auto startTime = std::chrono::high_resolution_clock::now();

    auto currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<f32, std::chrono::seconds::period>(currentTime - startTime).count();

    // NOTE: Y axis inversion in projection matrix
    UBO_MVP mvp{ .model = glm::rotate(glm::mat4(1.0f), duration * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                 .view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                 .projection = glm::perspective(glm::radians(45.0f), f32(m_swapchainExtent.width) / m_swapchainExtent.height, 0.1f, 10.0f) };

    mvp.projection[1][1] *= -1.0f;

    auto data = m_device.mapMemory(m_uniformBuffersMemory[imageIndex], 0, sizeof(mvp));
    std::memcpy(data, &mvp, sizeof(mvp));
    m_device.unmapMemory(m_uniformBuffersMemory[imageIndex]);
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
bool _isDeviceSuitable(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
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
QueueFamilyIndices _getRequiredQueueFamilies(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
{
    QueueFamilyIndices indices;

    // NOTE: This can replace indices for the queues already found, may be add smth like:
    //   if (indices.graphics_family.has_value() == false && queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
    for (ui32 i = 0; const auto & queueFamily : device.getQueueFamilyProperties()) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }
        if (device.getSurfaceSupportKHR(i, surface)) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }
        ++i;
    }

    return indices;
}

bool _checkPhysicalDeviceExtensionSupport(const vk::PhysicalDevice& device)
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


SwapchainSupportDetails _querySwapchainSupport(const vk::PhysicalDevice& device, const vk::SurfaceKHR& surface)
{
    return { .formats = device.getSurfaceFormatsKHR(surface),
             .presentModes = device.getSurfacePresentModesKHR(surface),
             .capabilities = device.getSurfaceCapabilitiesKHR(surface) };
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
    /*auto mode = std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eMailbox);
    if (mode != availablePresentModes.end())
        return *mode;

    return vk::PresentModeKHR::eFifo;*/
    return vk::PresentModeKHR::eImmediate;
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

vk::UniqueShaderModule _createShaderModule(const std::vector<char>& shaderCode, const vk::Device& device)
{
    // NOTE: May be read shader file as 'ui32' instead of 'char'
    vk::ShaderModuleCreateInfo shaderModuleInfo{ .codeSize = shaderCode.size(),
                                                 .pCode = reinterpret_cast<const ui32*>(shaderCode.data()) };

    return device.createShaderModuleUnique(shaderModuleInfo);
}


ui32 _findMemoryTypeIndex(const vk::PhysicalDevice& physicalDevice, const ui32 memoryTypeBits, const vk::MemoryPropertyFlags properties)
{
    auto memoryProperties = physicalDevice.getMemoryProperties();

    for (ui32 i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if (memoryTypeBits & (1 << i) && properties == (memoryProperties.memoryTypes[i].propertyFlags & properties)) {
            return i;
        }
    }

    throw std::runtime_error("_findMemoryType(): Failed to find suitable memory type!");
}

// NOTE: This function is called many times, and every time it calls getBufferMemoryRequirements() and findMemoryTypeIndex,
//  which is most likely redundant and I should just store this memory things
void _createBuffer(const vk::PhysicalDevice& physicalDevice, const vk::Device& device,
                   const vk::DeviceSize bufferSize,
                   const vk::BufferUsageFlags usage, const vk::MemoryPropertyFlags properties,
                   vk::Buffer& buffer, vk::DeviceMemory& bufferMemory)
{
    vk::BufferCreateInfo bufferInfo{ .size = bufferSize,
                                     .usage = usage,
                                     .sharingMode = vk::SharingMode::eExclusive };

    buffer = device.createBuffer(bufferInfo);

    const auto memoryRequirements = device.getBufferMemoryRequirements(buffer);
    const auto memoryTypeIndex = _findMemoryTypeIndex(physicalDevice, memoryRequirements.memoryTypeBits, properties);

    vk::MemoryAllocateInfo allocateInfo{ .allocationSize = memoryRequirements.size,
                                         .memoryTypeIndex = memoryTypeIndex };

    bufferMemory = device.allocateMemory(allocateInfo);
    device.bindBufferMemory(buffer, bufferMemory, 0);
}

void _copyBuffer(const vk::CommandPool& commandPool, const vk::Device& device, const vk::Queue& queue,
                 vk::Buffer& source, vk::Buffer& destination, vk::DeviceSize size)
{
    vk::CommandBufferAllocateInfo allocateInfo{ .commandPool = commandPool,
                                                .level = vk::CommandBufferLevel::ePrimary,
                                                .commandBufferCount = 1 };

    vk::CommandBuffer commandBuffer;
    device.allocateCommandBuffers(&allocateInfo, &commandBuffer);

    vk::CommandBufferBeginInfo beginInfo{ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit };

    commandBuffer.begin(beginInfo);
    {
        vk::BufferCopy copyRegion{ .size = size };
        commandBuffer.copyBuffer(source, destination, 1, &copyRegion);
    }
    commandBuffer.end();

    vk::SubmitInfo submitInfo{ .commandBufferCount = 1,
                              .pCommandBuffers = &commandBuffer };

    queue.submit(1, &submitInfo, nullptr);
    queue.waitIdle();

    device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}
