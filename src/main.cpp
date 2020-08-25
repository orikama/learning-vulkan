//#define GLFW_INCLUDE_VULKAN
// GLFW_VULKAN_STATIC ???
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <optional>
#include <cstdint>
#include <cstring> // std::strcmp


using i32 = std::int32_t;
using ui32 = std::uint32_t;


const ui32 kWindowWidth = 800;
const ui32 kWindowHeight = 600;

const std::vector<const char*> kDeviceExtensions{ VK_KHR_SWAPCHAIN_EXTENSION_NAME };
const std::vector<const char*> kValidationLayers{ "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
    constexpr bool kEnableValidationLayers = false;
#else
    constexpr bool kEnableValidationLayers = true;
#endif


static VKAPI_ATTR VkBool32 VKAPI_CALL
_debug_message_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                        VkDebugUtilsMessageTypeFlagsEXT message_type,
                        const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                        void* /*user_data*/)
{
    std::cerr << "validation layer: " << callback_data->pMessage << std::endl;

    return VK_FALSE;
}

static VKAPI_ATTR VkResult VKAPI_CALL
vkCreateDebugUtilsMessengerEXT(VkInstance instance,
                               const VkDebugUtilsMessengerCreateInfoEXT* create_info,
                               const VkAllocationCallbacks* alloc_callbacks,
                               VkDebugUtilsMessengerEXT* debug_messenger)
{
    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>
        (vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func == nullptr)
        return VK_ERROR_EXTENSION_NOT_PRESENT;

    return func(instance, create_info, alloc_callbacks, debug_messenger);
}

static VKAPI_ATTR void VKAPI_CALL
vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
                                VkDebugUtilsMessengerEXT messenger,
                                const VkAllocationCallbacks* pAllocator)
{
    auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>
        (vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
    if (func != nullptr)
        func(instance, messenger, pAllocator);
}


class TriangleApp
{
public:
    ~TriangleApp()
    {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    // NOTE: I should probably move _init_*() to constructor
    void run()
    {
        _init_window();
        _init_vulkan();
        _main_loop();
    }

private:
    GLFWwindow* m_window;

    vk::UniqueInstance m_vkInstance;
    vk::UniqueDebugUtilsMessengerEXT m_debug_messenger;
    vk::UniqueSurfaceKHR m_surface;

    vk::PhysicalDevice m_physical_device;
    vk::UniqueDevice m_device;

    vk::Queue m_graphics_queue;
    vk::Queue m_present_queue;

    // NOTE: Can be removed?
    struct QueueFamilyIndices
    {
        std::optional<ui32> graphics_family;
        std::optional<ui32> present_family;

        bool is_complete()
        {
            return graphics_family.has_value() && present_family.has_value();
        }
    };

    struct SwapChainSupportDetails
    {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> present_modes;
    };


    // Main methods

    void _init_window()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan", nullptr, nullptr);
    }

    void _init_vulkan()
    {
        _create_vkInstance();
        _setup_debug_messenger();
        _create_surface();
        _pick_physical_device();
        _create_logical_device();
    }

    void _main_loop()
    {
        while (glfwWindowShouldClose(m_window) == false) {
            glfwPollEvents();
        }
    }

    // Helper methods

    void _create_vkInstance()
    {
        if (kEnableValidationLayers && !_check_validation_layers_support()) {
            throw std::runtime_error("Validation layers requested, but not available!");
        }

        auto extensions = _get_required_extensions();

        vk::ApplicationInfo appInfo{ .pApplicationName = "VkTriangle",
                                     .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
                                     .pEngineName = "No Engine",
                                     .engineVersion = VK_MAKE_VERSION(0, 1, 0),
                                     .apiVersion = VK_API_VERSION_1_2 };

        vk::InstanceCreateInfo createInfo{ .pApplicationInfo = &appInfo,
                                           .enabledExtensionCount = static_cast<ui32>(extensions.size()),
                                           .ppEnabledExtensionNames = extensions.data() };

        auto debug_create_info = _make_debugUtilsMessengerCreateInfo();
        if (kEnableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<ui32>(kValidationLayers.size());
            createInfo.ppEnabledLayerNames = kValidationLayers.data();
            createInfo.pNext = &debug_create_info;
        } else {
            createInfo.enabledLayerCount = 0;
        }

        m_vkInstance = vk::createInstanceUnique(createInfo);
    }

    void _setup_debug_messenger()
    {
        if (kEnableValidationLayers == false)
            return;

        auto create_info = _make_debugUtilsMessengerCreateInfo();
        m_debug_messenger = m_vkInstance->createDebugUtilsMessengerEXTUnique(create_info);
    }

    void _create_surface()
    {
        // NOTE: Don't know if there is a way to make it without 'tmp'
        VkSurfaceKHR tmp;

        if (glfwCreateWindowSurface(m_vkInstance.get(), m_window, nullptr, &tmp) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create a window surface!");
        }
        m_surface = vk::UniqueSurfaceKHR(tmp, m_vkInstance.get());
    }

    void _pick_physical_device()
    {
        auto physical_devices = m_vkInstance->enumeratePhysicalDevices();
        // NOTE: Do I need to check this? Or vulkan.hpp will do this for me?
        if (physical_devices.size() == 0) {
            throw std::runtime_error("Failed to find GPUs with Vulkan support!");
        }

        for (const auto& device : physical_devices) {
            if (_is_device_suitable(device)) {
                m_physical_device = device;
                break;
            }
        }

        if (m_physical_device == vk::PhysicalDevice()) {
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    void _create_logical_device()
    {
        QueueFamilyIndices indices = _find_queue_families(m_physical_device);

        std::unordered_set<ui32> unique_queue_families{ indices.graphics_family.value(),
                                                        indices.present_family.value() };
        std::vector<vk::DeviceQueueCreateInfo> queue_create_infos;
        queue_create_infos.reserve(unique_queue_families.size());

        const float queue_priority = 1.0f; // NOTE: same for every queue
        for (ui32 queue_family : unique_queue_families) {
            queue_create_infos.push_back(vk::DeviceQueueCreateInfo{ .queueFamilyIndex = queue_family,
                                                                    .queueCount = 1,
                                                                    .pQueuePriorities = &queue_priority });
        }

        vk::PhysicalDeviceFeatures device_features{}; // NOTE: empty for now

        // DIFFERENCE: Skipped enabling validation layers for device, since there is no need to do that in modern Vulkan

        vk::DeviceCreateInfo create_info{ .queueCreateInfoCount = static_cast<ui32>(queue_create_infos.size()),
                                          .pQueueCreateInfos = queue_create_infos.data(),
                                          .enabledExtensionCount = static_cast<ui32>(kDeviceExtensions.size()),
                                          .ppEnabledExtensionNames = kDeviceExtensions.data(),
                                          .pEnabledFeatures = &device_features };

        m_device = m_physical_device.createDeviceUnique(create_info);
        // NOTE: m_graphics_queue and m_present_queue can hold the same value
        m_graphics_queue = m_device->getQueue(indices.graphics_family.value(), 0);
        m_present_queue = m_device->getQueue(indices.present_family.value(), 0);
    }

    vk::DebugUtilsMessengerCreateInfoEXT _make_debugUtilsMessengerCreateInfo()
    {
        vk::DebugUtilsMessageSeverityFlagsEXT severity{
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
            | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError };
        vk::DebugUtilsMessageTypeFlagsEXT type{
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
            | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
            | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance };
        vk::DebugUtilsMessengerCreateInfoEXT create_info{ .messageSeverity = severity,
                                                          .messageType = type,
                                                          .pfnUserCallback = _debug_message_callback };
        return create_info;
    }

    SwapChainSupportDetails _query_swap_chain_support(vk::PhysicalDevice device)
    {
        SwapChainSupportDetails details{ .capabilities = m_physical_device.getSurfaceCapabilitiesKHR(m_surface.get()),
                                         .formats = m_physical_device.getSurfaceFormatsKHR(m_surface.get()),
                                         .present_modes = m_physical_device.getSurfacePresentModesKHR(m_surface.get()) };
        return details;
    }

    bool _is_device_suitable(vk::PhysicalDevice device)
    {
        bool is_queue_families_supported = _find_queue_families(device).is_complete();
        bool is_extensions_supported = _check_device_extension_support(device);
        bool is_swap_chain_adequate = false;
        if (is_extensions_supported) {
            auto swap_chain_support = _query_swap_chain_support(device);
            // NOTE: Make a SwapChainSupportDetails method
            is_swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
        }

        return is_queue_families_supported && is_extensions_supported && is_swap_chain_adequate;
    }

    bool _check_device_extension_support(vk::PhysicalDevice device)
    {
        auto available_extensions = device.enumerateDeviceExtensionProperties();
        
        return std::all_of(kDeviceExtensions.begin(), kDeviceExtensions.end(),
                           [&available_extensions](const char* required) {
                               return std::find_if(available_extensions.begin(), available_extensions.end(),
                                                   [&required](const vk::ExtensionProperties& available) {
                                                       return std::strcmp(required, available.extensionName) == 0;
                                                   }) != available_extensions.end();
                           });
    }

    QueueFamilyIndices _find_queue_families(vk::PhysicalDevice device)
    {
        QueueFamilyIndices indices;

        auto queue_families = device.getQueueFamilyProperties();

        // NOTE: This can replace indices for the queues already found, may be add smth like:
        //   if (indices.graphics_family.has_value() == false && queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
        for (ui32 i = 0; const auto & queue_family : queue_families) {
            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphics_family = i;
            }
            if (device.getSurfaceSupportKHR(i, m_surface.get())) {
                indices.present_family = i;
            }

            if (indices.is_complete()) {
                break;
            }
            ++i;
        }

        return indices;
    }

    std::vector<const char*> _get_required_extensions()
    {
        ui32 glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
        if (kEnableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    bool _check_validation_layers_support()
    {
        auto available_layers = vk::enumerateInstanceLayerProperties();

        return std::all_of(kValidationLayers.begin(), kValidationLayers.end(),
                           [&available_layers](const char* required) {
                               return std::find_if(available_layers.begin(), available_layers.end(),
                                                   [&required](const vk::LayerProperties& available) {
                                                       return std::strcmp(required, available.layerName) == 0;
                                                   }) != available_layers.end();
                           });
    }
};


int main()
{
    TriangleApp app;

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
