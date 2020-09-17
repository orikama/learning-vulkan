#pragma once

#include "core.hpp"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan.hpp>

#include "Window.hpp"

#include <iostream> // TODO: Remove


// TODO: Remove this shit from here
static VKAPI_ATTR VkBool32 VKAPI_CALL
_debugMessageCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                      VkDebugUtilsMessageTypeFlagsEXT message_type,
                      const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                      void* /*user_data*/)
{
    std::cerr << "Validation layer: " << callback_data->pMessage << std::endl;

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


namespace vulkan
{

class VkBackend
{
public:
    VkBackend() = default;
    //VkBackend(ui32 apiVersion);
    
    /*VkBackend(VkBackend&& tmp) noexcept;
    VkBackend& operator=(VkBackend&& tmp) noexcept;*/

    VkBackend(const VkBackend&) = delete;
    VkBackend& operator=(const VkBackend&) = delete;

    void Init(const Window& window);

private:
    void _CreateVkInstance(ui32 apiVersion);
    void _SetupDebugMessenger();
    void _CreateSurface(GLFWwindow* windowHandle);
    void _SelectPhysicalDevice();
    void _CreateLogicalDeviceAndQueues();
    void _CreateSwapChain();

private:
    // TODO: There is probably no need to use unique handles, it's pretty easy to manage them manually anyway.
    //  From vulkan.hpp readme:
    //    Note that using vk::UniqueHandle comes at a cost since most deleters have to store the
    //    vk::AllocationCallbacks and parent handle used for construction because they are required for automatic destruction.

    vk::UniqueInstance                  m_vkInstance;

    // TODO: I should remove this on release build with preprocessor help,
    //  although probably with more optimization options enabled it will be removed.
    vk::UniqueDebugUtilsMessengerEXT    m_debugMessenger;

    vk::UniqueSurfaceKHR                m_surface;

    vk::PhysicalDevice                  m_physicalDevice;
    vk::UniqueDevice                    m_device;

    vk::Queue m_graphicsQueue;
    vk::Queue m_presentQueue;
};

}
