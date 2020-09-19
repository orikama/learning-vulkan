//#define GLFW_INCLUDE_VULKAN
// GLFW_VULKAN_STATIC ???
//#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
//#include <vulkan/vulkan.hpp>
//#include <GLFW/glfw3.h>


#include "VkBackend.hpp"
#include "Window.hpp"


constexpr ui32 kWindowWidth = 800;
constexpr ui32 kWindowHeight = 600;



class TriangleApp
{
public:
    TriangleApp()
    {
        m_window.Init(kWindowWidth, kWindowHeight, "Vulkan");
        m_vkBackend.Init(m_window);
    }

    ~TriangleApp()
    {
        m_window.Shutdown();
    }

    void run()
    {
        while (m_window.ShouldClose() == false) {
            m_window.PollEvents();
            m_vkBackend.DrawFrame();
        }
        m_vkBackend.WaitIdle();
    }

private:
    Window m_window;
    vulkan::VkBackend m_vkBackend;
};


int main()
{

    try {
        TriangleApp app;
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
