#pragma once

#include "core.hpp"

#include <GLFW/glfw3.h>


class Window
{
public:
    Window() = default;

    void Init(ui32 width, ui32 height, const char* title);
    void Shutdown();

    GLFWwindow* GetWindowHandle() const;

    i32 GetWidth() const;
    i32 GetHeight() const;

    void MainLoop();

private:
    GLFWwindow* m_window;

    ui32 m_width;
    ui32 m_height;
};
