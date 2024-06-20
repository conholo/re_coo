#include "application.h"
#include "../../renderer.h"
#include "renderer/scratch_renderer.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <glm/glm.hpp>
#include <chrono>

Application::Application()
{
    assert(!s_ApplicationInstance && "Application instance already exists.");
    s_ApplicationInstance = this;
}

Application::~Application() = default;

void Application::Run()
{
    RTRenderer renderer(m_Window, m_VulkanDevice);
    renderer.Initialize();
    auto currentTime = std::chrono::high_resolution_clock::now();
    m_Camera.SetPerspectiveProjection(glm::radians(50.0f), renderer.GetAspectRatio(), 0.1f, 100.0f);

    while(!m_Window.ShouldClose())
    {
        glfwPollEvents();

        auto newTime = std::chrono::high_resolution_clock::now();
        float frameTime = std::chrono::duration<float>(newTime - currentTime).count();
        currentTime = newTime;

        m_Camera.Tick(frameTime);
        renderer.Render(m_Camera);
    }

    vkDeviceWaitIdle(m_VulkanDevice.GetDevice());
}