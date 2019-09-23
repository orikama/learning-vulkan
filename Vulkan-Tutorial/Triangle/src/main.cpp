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
    void Run()
    {
        InitVulkan();
        MainLoop();
        Cleanup();
    }

private:
    void InitVulkan()
    {

    }

    void MainLoop()
    {

    }

    void Cleanup()
    {

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
