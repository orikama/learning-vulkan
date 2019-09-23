#include <iostream>
#include <stdexcept>
#include <functional>
#include <cstdlib>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/vec4.hpp>
#include <glm/mat4x4.hpp>

class HelloTriangleApplication
{
public:
    const int kWindowWidth = 800;
    const int kWindowHeight = 600;

    void Run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* m_window = nullptr;

    VkInstance m_vkInstance = nullptr;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        
        m_window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Vulkan", nullptr, nullptr);
    }

    void initVulkan()
    {
        createVkInstance();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        vkDestroyInstance(m_vkInstance, nullptr);

        glfwDestroyWindow(m_window);
        glfwTerminate();
    }

    void createVkInstance()
    {
        VkApplicationInfo applicationInfo = {};
        applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        applicationInfo.pApplicationName = "Hello Triangle";
        applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.pEngineName = "No Engine";
        applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        applicationInfo.apiVersion = VK_API_VERSION_1_0;

        uint32_t glfwExtensionCount = 0u;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        /*std::cout << "GLFW required extensions: " << glfwExtensionCount << '\n';
        for (uint32_t i = 0; i < glfwExtensionCount; ++i) {
            std::cout << glfwExtensions[i] << '\n';
        }*/

        VkInstanceCreateInfo createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &applicationInfo;
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0u;

        VkResult result = vkCreateInstance(&createInfo, nullptr, &m_vkInstance);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to create vulkan instance.");
        }
    }
};

int main()
{
    HelloTriangleApplication app;

    try {
        app.Run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
