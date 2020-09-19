#include "Window.hpp"

#include <stdexcept>


void Window::Init(ui32 width, ui32 height, const char* title)
{
    glfwInit();

    if (glfwVulkanSupported() == GLFW_FALSE) {
        throw std::runtime_error("Vulkan is not supported on the current hardware!");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    m_width = width;
    m_height = height;
    m_window = glfwCreateWindow(width, height, title, nullptr, nullptr);
}

void Window::Shutdown()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

GLFWwindow* Window::GetWindowHandle() const
{
    return m_window;
}

ui32 Window::GetWidth() const
{
    return m_width;
}

ui32 Window::GetHeight() const
{
    return m_height;
}

// NOTE: Questionable method
bool Window::ShouldClose() const
{
    return glfwWindowShouldClose(m_window);
}
// NOTE: Questionable method
void Window::PollEvents() const
{
    glfwPollEvents();
}
