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
    void Shutdown();

    void DrawFrame();
    // NOTE: Questionable method
    void WaitIdle() const;

private:
    void _CreateInstance(ui32 apiVersion);
    void _SetupDebugMessenger();
    void _CreateSurface(GLFWwindow* windowHandle);
    void _SelectPhysicalDevice();
    void _CreateLogicalDeviceAndQueues();
    void _CreateSwapchain(ui32 width, ui32 height);
    void _CreateImageViews();
    void _CreateRenderPass();
    void _CreateGraphicsPipeline();
    void _CreateFramebuffers();
    void _CreateCommandPool();
    void _CreateVertexBuffer();
    void _CreateCommandBuffers();
    void _CreateSyncPrimitives();

    void _CleanupSwapchain();
    //void _RecreateSwapchain();

private:
    ui64 m_frameCounter;
    ui32 m_currentFrameData;


    vk::Instance                    m_instance;

    // TODO: I should remove this on release build with preprocessor help,
    //  although probably with more optimization options enabled it will be removed.
    vk::DebugUtilsMessengerEXT      m_debugMessenger;

    vk::SurfaceKHR                  m_surface;

    vk::PhysicalDevice              m_physicalDevice;
    vk::Device                      m_device;

    vk::Queue                       m_graphicsQueue;
    vk::Queue                       m_presentQueue;

    vk::SwapchainKHR                m_swapchain;
    vk::Format                      m_swapchainFormat;
    vk::Extent2D                    m_swapchainExtent;


    std::vector<vk::Image>          m_swapchainImages;
    std::vector<vk::ImageView>      m_swapchainImageViews;
    std::vector<vk::Framebuffer>    m_framebuffers;


    vk::RenderPass                  m_renderPass;
    // TODO: Move this and all stuff about shaders to its own class, as done in DOOM3 ?
    vk::PipelineLayout              m_pipelineLayout;
    vk::Pipeline                    m_pipeline;


    vk::CommandPool                 m_commandPool;
    std::vector<vk::CommandBuffer>  m_commandBuffers;
    std::vector<vk::Semaphore>      m_imageAvailableSemaphores;
    std::vector<vk::Semaphore>      m_renderFinishedSemaphores;
    std::vector<vk::Fence>          m_inFlightFences;

    vk::Buffer                      m_vertexBuffer;
    vk::DeviceMemory                m_vertexBufferMemory;
};

}
