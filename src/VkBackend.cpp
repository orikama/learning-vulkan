#include "VkBackend.hpp"

#include "Window.hpp"

#include <stdexcept> // std::runtime_error
#include <algorithm>
#include <optional>
#include <cstring> // std::strcmp
#include <limits>
#include <unordered_set>
#include <vector>


#ifdef NDEBUG
    constexpr bool kEnableValidationLayers = false;
#else
    constexpr bool kEnableValidationLayers = true;
#endif

const std::vector<const char*> kValidationLayers{ "VK_LAYER_KHRONOS_validation" };
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

struct SwapChainSupportDetails
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
auto _querySwapChainSupport(const vk::PhysicalDevice device,
                            const vk::UniqueSurfaceKHR& surface)            -> SwapChainSupportDetails;


namespace vulkan
{

void VkBackend::Init(const Window& window)
{
    _CreateVkInstance(VK_API_VERSION_1_2);
    _SetupDebugMessenger();
    _CreateSurface(window.GetWindowHandle());
    _SelectPhysicalDevice();
    _CreateLogicalDeviceAndQueues();
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
    const QueueFamilyIndices indices = _getRequiredQueueFamilies(m_physicalDevice, m_surface);

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

void VkBackend::_CreateSwapChain()
{
    auto swapChainSupport = _querySwapChainSupport(m_physicalDevice, m_surface);

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
        auto swapChainSupport = _querySwapChainSupport(device, surface);
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

SwapChainSupportDetails _querySwapChainSupport(const vk::PhysicalDevice device, const vk::UniqueSurfaceKHR& surface)
{
    SwapChainSupportDetails details{ .formats = device.getSurfaceFormatsKHR(surface.get()),
                                     .presentModes = device.getSurfacePresentModesKHR(surface.get()),
                                     .capabilities = device.getSurfaceCapabilitiesKHR(surface.get()) };
    return details;
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
    /*for (const auto presentMode : availablePresentModes) {
        if (presentMode == vk::PresentModeKHR::eMailbox) {
            return presentMode;
        }
    }*/
    // TODO: Rewrite with ranges?
    auto mode = std::find(availablePresentModes.begin(), availablePresentModes.end(), vk::PresentModeKHR::eMailbox);
    if (mode != availablePresentModes.end())
        return *mode;

    return vk::PresentModeKHR::eFifo;
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

