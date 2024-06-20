#include <array>
#include "renderer.h"
#include "renderer/vulkan/vulkan_utils.h"
#include "scene/scene.h"
#include <random>
#include <ctime>
#include <sstream>

Renderer::Renderer(Window& windowRef, VulkanDevice& deviceRef)
    :m_WindowRef(windowRef), m_DeviceRef(deviceRef)
{
    RecreateSwapchain();
    AllocateCommandBuffers();
    CreateFramebuffers();
    CreateAccumulationFramebuffer();
    CreateSynchronizationPrimitives();

    SetupCompositeDescriptors();
    SetupOffscreenDescriptors();
    SetupGlobalDescriptors();

    CreateOffscreenGraphicsPipelineLayout();
    CreateOffscreenGraphicsPipeline();

    CreateAccumulationGraphicsPipelineLayout();
    CreateAccumulationGraphicsPipeline();

    CreateCompositionGraphicsPipelineLayout();
    CreateCompositionGraphicsPipeline();

    m_QueueFamily = m_DeviceRef.FindPhysicalQueueFamilies();

    Sphere sphereA
    {
        .Position_Radius{-2.0f, 1.0f, 0.0f, 1.0f},
        .Material
        {
            .Color_Smoothness{0.9f, 0.0f, 0.1f, 0.0f},
            .EmissionColor_Strength{0.0f, 0.0f, 0.0f, 0.0f},
            .SpecularColor_Probability{1.0f, 1.0f, 1.0f, 0.5f},
        }
    };
    Sphere sphereB
    {
        .Position_Radius{2.5f, 1.0f, 0.0f, 2.0f},
        .Material
        {
            .Color_Smoothness{0.1f, 0.8f, 0.1f, 1.0f},
            .EmissionColor_Strength{0.0f, 0.0f, 0.0f, 0.0f},
            .SpecularColor_Probability{1.0f, 1.0f, 1.0f, 0.9f},
        }
    };
    Sphere emissiveSphereA
    {
        .Position_Radius{0.0f, 5.0f, 0.0f, 0.5f},
        .Material
        {
            .Color_Smoothness{0.0f, 0.0f, 0.0f, 0.0f},
            .EmissionColor_Strength{1.0f, 1.0f, 1.0f, 1.0f},
            .SpecularColor_Probability{0.0f, 0.0f, 0.0f, 0.0f},
        }
    };

    std::vector<Sphere> spheres { sphereA, sphereB, emissiveSphereA };
    uint32_t sphereCount = spheres.size();

    VulkanBuffer stagingBuffer {
        m_DeviceRef,
        sizeof(Sphere),
        sphereCount,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    };

    stagingBuffer.Map();
    stagingBuffer.WriteToBuffer(spheres.data());

    m_SphereSSBOs.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);

    // Copy sphere data to all storage buffers
    for (size_t i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_SphereSSBOs[i] = std::make_unique<VulkanBuffer>(
                m_DeviceRef,
                sizeof(Sphere),
                spheres.size(),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        m_DeviceRef.CopyBuffer(stagingBuffer.GetBuffer(), m_SphereSSBOs[i]->GetBuffer(), stagingBuffer.GetBufferSize());
    }
}

Renderer::~Renderer()
{

}

void Renderer::RecreateSwapchain()
{
    auto extent = m_WindowRef.GetExtent();
    while (extent.width == 0 || extent.height == 0)
    {
        extent = m_WindowRef.GetExtent();
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(m_DeviceRef.GetDevice());

    if (m_Swapchain == nullptr)
    {
        m_Swapchain = std::make_unique<VulkanSwapchain>(m_DeviceRef, extent);
    }
    else
    {
        std::shared_ptr<VulkanSwapchain> oldSwapChain = std::move(m_Swapchain);
        m_Swapchain = std::make_unique<VulkanSwapchain>(m_DeviceRef, extent, oldSwapChain);

        if (!oldSwapChain->CompareSwapchainFormats(*m_Swapchain))
        {
            throw std::runtime_error("Swap chain image(or depth) format has changed!");
        }
    }
}

void Renderer::Render(Camera& cameraRef)
{
    GlobalUbo ubo{};
    ubo.Projection = cameraRef.GetProjection();
    ubo.View = cameraRef.GetView();
    ubo.InvView = cameraRef.GetInvView();
    ubo.InvProjection = cameraRef.GetInvProjection();
    ubo.CameraPosition = glm::vec4(cameraRef.GetPosition(), 0.0f);
    constexpr int RaysPerPixel = 1;
    ubo.ScreenResolution_NumRaysPerPixel_FrameNumber = glm::ivec4(
            m_Swapchain->GetWidth(),
            m_Swapchain->GetHeight(),
            RaysPerPixel,
            m_FrameCounter);

    m_GlobalUBOs[m_CurrentFrameIndex]->WriteToBuffer(&ubo);
    m_GlobalUBOs[m_CurrentFrameIndex]->Flush();

    // Submit Offscreen Command Buffer
    {
        BeginOffscreenRenderPass(m_OffscreenCommandBuffers[m_CurrentFrameIndex], *m_SphereSSBOs[m_CurrentFrameIndex]);
        RenderSceneOffscreen(m_OffscreenCommandBuffers[m_CurrentFrameIndex], m_GlobalDescriptorSets[m_CurrentFrameIndex], *m_SphereSSBOs[m_CurrentFrameIndex]);
        EndSwapchainRenderPass(m_OffscreenCommandBuffers[m_CurrentFrameIndex]);

        VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT };

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_OffscreenCommandBuffers[m_CurrentFrameIndex];

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_GraphicsSemaphores[m_CurrentFrameIndex];
        submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;

        // Signal ready with offscreen semaphore.
        submitInfo.pSignalSemaphores = &m_OffscreenRenderFinishedSemaphores[m_CurrentFrameIndex];
        submitInfo.signalSemaphoreCount = 1;
        VK_CHECK_RESULT(vkQueueSubmit(m_DeviceRef.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    }

    // Submit Offscreen Command Buffer
    {
        BeginAccumulationPass(m_AccumulationCommandBuffers[m_CurrentFrameIndex], *m_SphereSSBOs[m_CurrentFrameIndex]);
        RenderAccumulationPass(m_AccumulationCommandBuffers[m_CurrentFrameIndex], m_GlobalDescriptorSets[m_CurrentFrameIndex], *m_SphereSSBOs[m_CurrentFrameIndex]);
        EndAccumulationPass(m_AccumulationCommandBuffers[m_CurrentFrameIndex]);

        VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT };

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_OffscreenCommandBuffers[m_CurrentFrameIndex];

        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_GraphicsSemaphores[m_CurrentFrameIndex];
        submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;

        // Signal ready with offscreen semaphore.
        submitInfo.pSignalSemaphores = &m_OffscreenRenderFinishedSemaphores[m_CurrentFrameIndex];
        submitInfo.signalSemaphoreCount = 1;
        VK_CHECK_RESULT(vkQueueSubmit(m_DeviceRef.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
    }

    // Swapchain Acquisition/Submission/Presentation
    {
        //Acquisition
        {
            vkWaitForFences(m_DeviceRef.GetDevice(), 1, &m_WaitFences[m_CurrentFrameIndex], VK_TRUE, std::numeric_limits<uint64_t>::max());

            auto result = m_Swapchain->AcquireNextImage(&m_CurrentBufferIndex, m_PresentCompleteSemaphores[m_CurrentFrameIndex]);

            if(result == VK_ERROR_OUT_OF_DATE_KHR)
            {
                RecreateSwapchain();
            }
            if(result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
                throw std::runtime_error("Failed to acquire swap chain image!");
        }

        // Submit Composition Command Buffer
        {
            BeginSwapchainRenderPass(m_DrawCommandBuffers[m_CurrentFrameIndex], m_CurrentBufferIndex);
            RenderComposition(m_DrawCommandBuffers[m_CurrentFrameIndex], m_GlobalDescriptorSets[m_CurrentFrameIndex]);
            EndSwapchainRenderPass(m_DrawCommandBuffers[m_CurrentFrameIndex]);

            if (m_ImagesInFlightFences[m_CurrentBufferIndex] != VK_NULL_HANDLE)
            {
                VK_CHECK_RESULT(vkWaitForFences(m_DeviceRef.GetDevice(), 1, &m_ImagesInFlightFences[m_CurrentBufferIndex], VK_TRUE, UINT64_MAX));
            }
            m_ImagesInFlightFences[m_CurrentBufferIndex] = m_WaitFences[m_CurrentFrameIndex];
            vkResetFences(m_DeviceRef.GetDevice(), 1, &m_WaitFences[m_CurrentFrameIndex]);

            VkPipelineStageFlags graphicsWaitStageMasks[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
            VkSemaphore graphicsWaitSemaphores[] = { m_OffscreenRenderFinishedSemaphores[m_CurrentFrameIndex], m_PresentCompleteSemaphores[m_CurrentFrameIndex] };
            VkSemaphore graphicsSignalSemaphores[] = { m_GraphicsSemaphores[m_CurrentFrameIndex], m_RenderCompleteSemaphores[m_CurrentFrameIndex] };

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &m_DrawCommandBuffers[m_CurrentFrameIndex];

            submitInfo.waitSemaphoreCount = 2;
            submitInfo.pWaitSemaphores = graphicsWaitSemaphores;
            submitInfo.pWaitDstStageMask = graphicsWaitStageMasks;

            submitInfo.signalSemaphoreCount = 2;
            submitInfo.pSignalSemaphores = graphicsSignalSemaphores;
            submitInfo.pNext = nullptr;
            VK_CHECK_RESULT(vkQueueSubmit(m_DeviceRef.GetGraphicsQueue(), 1, &submitInfo, m_WaitFences[m_CurrentFrameIndex]));
        }

        // Presentation
        {
            auto result = m_Swapchain->Present(
                    m_DeviceRef.GetPresentQueue(),
                    m_CurrentBufferIndex,
                    m_RenderCompleteSemaphores[m_CurrentFrameIndex]);

            if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_WindowRef.WasWindowResized())
            {
                m_WindowRef.ResetWindowResizedFlag();
                RecreateSwapchain();
                OnSwapchainResized(m_Swapchain->GetWidth(), m_Swapchain->GetHeight());
            }
            else if (result != VK_SUCCESS)
            {
                throw std::runtime_error("Failed to present swapchain image!");
            }
            VK_CHECK_RESULT(vkQueueWaitIdle(m_DeviceRef.GetGraphicsQueue()));
        }
    }

    m_CurrentFrameIndex = (m_CurrentFrameIndex + 1) % VulkanSwapchain::MAX_FRAMES_IN_FLIGHT;
    m_FrameCounter++;
}

void Renderer::BeginSwapchainRenderPass(VkCommandBuffer drawCommandBuffer, uint32_t bufferIndex)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(drawCommandBuffer, &beginInfo));

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_Swapchain->GetRenderPass();
    renderPassInfo.framebuffer = m_Swapchain->GetFrameBuffer(bufferIndex);

    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_Swapchain->GetSwapchainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {0.01f, 0.01f, 0.01f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(drawCommandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_Swapchain->GetSwapchainExtent().width);
    viewport.height = static_cast<float>(m_Swapchain->GetSwapchainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, m_Swapchain->GetSwapchainExtent()};
    vkCmdSetViewport(drawCommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(drawCommandBuffer, 0, 1, &scissor);
}

void Renderer::RenderComposition(VkCommandBuffer commandBuffer, VkDescriptorSet globalSet)
{
    m_CompositeDescriptorPool->ResetPool();
    m_CompositionGraphicsPipeline->Bind(commandBuffer);

    vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_CompositionGraphicsPipelineLayout,
            0,
            1,
            &globalSet,
            0,
            nullptr);

    VkDescriptorImageInfo colorAttachment = m_Framebuffers->GetDescriptorImageInfoForAttachment(0, m_FramebufferColorSampler);

    VkDescriptorSet compositeDescriptorSet;
    VulkanDescriptorWriter(*m_CompositeDescriptorSetLayout, *m_CompositeDescriptorPool)
            .WriteImage(0, &colorAttachment)
            .Build(compositeDescriptorSet);

    vkCmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_CompositionGraphicsPipelineLayout,
            1,
            1,
            &compositeDescriptorSet,
            0,
            nullptr);

    // Final composition
    // This is done by simply drawing a full screen quad
    // The fragment shader then samples from the fbo attachment
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
}

void Renderer::EndSwapchainRenderPass(VkCommandBuffer drawCommandBuffer)
{
    vkCmdEndRenderPass(drawCommandBuffer);
    vkEndCommandBuffer(drawCommandBuffer);
}

void Renderer::RenderOffscreen()
{
    uint64_t currIndex = m_FrameCounter % 3;
    uint64_t prevIndex = (m_FrameCounter + 1) % 3;
    uint64_t nextIndex = (m_FrameCounter + 2) % 3;


}

void Renderer::BeginOffscreenRenderPass(VkCommandBuffer offscreenCommandBuffer, VulkanBuffer& buffer)
{
    // Output of this pass is writing to all attachments for the offscreen FBO.
    // Clear values for all attachments written in the fragment shader

    // Begin recording the offscreen command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_Framebuffers->GetRenderPass();
    renderPassBeginInfo.framebuffer = m_Framebuffers->GetFramebuffer();
    renderPassBeginInfo.renderArea.extent.width = m_Framebuffers->GetWidth();
    renderPassBeginInfo.renderArea.extent.height = m_Framebuffers->GetHeight();
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    VK_CHECK_RESULT(vkBeginCommandBuffer(offscreenCommandBuffer, &beginInfo));

    vkCmdBeginRenderPass(offscreenCommandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_Framebuffers->GetWidth());
    viewport.height = static_cast<float>(m_Framebuffers->GetHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, renderPassBeginInfo.renderArea.extent};
    vkCmdSetViewport(offscreenCommandBuffer, 0, 1, &viewport);
    vkCmdSetScissor(offscreenCommandBuffer, 0, 1, &scissor);
}

void Renderer::RenderSceneOffscreen(VkCommandBuffer offscreenCommandBuffer, VkDescriptorSet globalSet, VulkanBuffer& sphereBuffer)
{
    m_OffscreenDescriptorPool->ResetPool();
    m_OffscreenGraphicsPipeline->Bind(offscreenCommandBuffer);

    vkCmdBindDescriptorSets(
            offscreenCommandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_OffscreenGraphicsPipelineLayout,
            0,
            1,
            &globalSet,
            0,
            nullptr);

    VkDescriptorSet bufferSet;
    VkDescriptorBufferInfo info = sphereBuffer.DescriptorInfo();
    VulkanDescriptorWriter(*m_MainRTPassDescriptorSetLayout, *m_OffscreenDescriptorPool)
            .WriteBuffer(0, &info)
            .Build(bufferSet);

    vkCmdBindDescriptorSets(
            offscreenCommandBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_OffscreenGraphicsPipelineLayout,
            1,
            1,
            &bufferSet,
            0,
            nullptr);

    vkCmdDraw(offscreenCommandBuffer, 3, 1, 0, 0);
}

void Renderer::EndOffscreenRenderPass(VkCommandBuffer offscreenCommandBuffer)
{
    vkCmdEndRenderPass(offscreenCommandBuffer);
    VK_CHECK_RESULT(vkEndCommandBuffer(offscreenCommandBuffer));
}

/*
 * Resources Allocation and Initialization Begin
 */

void Renderer::SetupGlobalDescriptors()
{
    m_GlobalSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: Global UBO
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT)
            .Build();

    m_GlobalDescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .Build();

    m_GlobalUBOs.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for (auto& uboBuffer : m_GlobalUBOs)
    {
        uboBuffer = std::make_unique<VulkanBuffer>(
                m_DeviceRef,
                sizeof(GlobalUbo),
                1,
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        uboBuffer->Map();
    }

    m_GlobalDescriptorSets.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);

    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        auto bufferInfo = m_GlobalUBOs[i]->DescriptorInfo();
        VulkanDescriptorWriter(*m_GlobalSetLayout, *m_GlobalDescriptorPool)
                .WriteBuffer(0, &bufferInfo)
                .Build(m_GlobalDescriptorSets[i]);
    }
}

void Renderer::SetupOffscreenDescriptors()
{
    m_MainRTPassDescriptorSetLayout =
            VulkanDescriptorSetLayout::Builder(m_DeviceRef)
                    // Binding 0: SSBO for Spheres
                    .AddBinding(
                            0,
                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            VK_SHADER_STAGE_FRAGMENT_BIT)
                    .Build();

    m_OffscreenDescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .Build();
}

void Renderer::SetupAccumulationDescriptors()
{
    m_AccumulationDescriptorSetLayout =
            VulkanDescriptorSetLayout::Builder(m_DeviceRef)
                    // Binding 0: FBO Attachment 0 (Previous) Color Sampler from Offscreen Pass
                    .AddBinding(
                            0,
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            VK_SHADER_STAGE_FRAGMENT_BIT)
                    // Binding 1: FBO Attachment 1 (Current) Color Sampler from Offscreen Pass
                    .AddBinding(
                            1,
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            VK_SHADER_STAGE_FRAGMENT_BIT)
                    .Build();

    m_AccumulationDescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 2)
            .Build();
}

void Renderer::SetupCompositeDescriptors()
{
    m_CompositeDescriptorSetLayout =
            VulkanDescriptorSetLayout::Builder(m_DeviceRef)
                    // Binding 0: FBO Attachment Color Sampler from Accumulation Pass
                    .AddBinding(
                            0,
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            VK_SHADER_STAGE_FRAGMENT_BIT)
                    .Build();

    m_CompositeDescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .Build();
}

void Renderer::CreateOffscreenGraphicsPipelineLayout()
{
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
            m_GlobalSetLayout->GetDescriptorSetLayout(),
            m_MainRTPassDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_OffscreenGraphicsPipelineLayout));
}

void Renderer::CreateOffscreenGraphicsPipeline()
{
    assert(m_OffscreenGraphicsPipelineLayout != nullptr && "Cannot create pipeline before pipeline layout!");

    VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
    VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

    VkPipelineColorBlendAttachmentState blendAttachmentState{};
    VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState{};
    pipelineColorBlendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
    pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;
    blendAttachmentState = pipelineColorBlendAttachmentState;

    pipelineConfig.ColorBlendInfo.attachmentCount = 1;
    pipelineConfig.ColorBlendInfo.pAttachments = &blendAttachmentState;

    pipelineConfig.RenderPass = m_Framebuffers->GetRenderPass();
    pipelineConfig.PipelineLayout = m_OffscreenGraphicsPipelineLayout;
    pipelineConfig.EmptyVertexInputState = true;
    m_OffscreenGraphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
            m_DeviceRef,
            "../assets/shaders/raytrace.vert.spv",
            "../assets/shaders/raytrace.frag.spv",
            pipelineConfig);
}

void Renderer::CreateAccumulationGraphicsPipelineLayout()
{
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_GlobalSetLayout->GetDescriptorSetLayout(),
        m_AccumulationDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_OffscreenGraphicsPipelineLayout));
}


void Renderer::CreateCompositionGraphicsPipelineLayout()
{
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_GlobalSetLayout->GetDescriptorSetLayout(),
        m_CompositeDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_CompositionGraphicsPipelineLayout));
}

void Renderer::CreateCompositionGraphicsPipeline()
{
    assert(m_CompositionGraphicsPipelineLayout != nullptr && "Cannot create pipeline before pipeline layout!");

    VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
    VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.RenderPass = m_Swapchain->GetRenderPass();
    pipelineConfig.PipelineLayout = m_CompositionGraphicsPipelineLayout;
    pipelineConfig.EmptyVertexInputState = true;
    m_CompositionGraphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
            m_DeviceRef,
            "../assets/shaders/fsq.vert.spv",
            "../assets/shaders/texture_display.frag.spv",
            pipelineConfig);
}

void Renderer::CreateFramebuffers()
{
    m_Framebuffers.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for(auto& framebuffer : m_Framebuffers)
    {
        VulkanFramebuffer::Attachment::Specification attachment0 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        };

        VulkanFramebuffer::Attachment::Specification attachment1 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        };

        VulkanFramebuffer::Attachment::Specification attachment2 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
        };

        std::vector<VulkanFramebuffer::Attachment::Specification> attachmentSpecs =
        {
            attachment0,
            attachment1,
            attachment2
        };

        // Define subpasses
        std::vector<VulkanFramebuffer::Subpass> subpasses;

        VulkanFramebuffer::Subpass rayTraceSubpass = {};
        rayTraceSubpass.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        rayTraceSubpass.ColorAttachments =
        {
            {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
        };
        subpasses.push_back(rayTraceSubpass);

        VulkanFramebuffer::Subpass accumPass = {};
        accumPass.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        accumPass.ColorAttachments =
        {
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
            {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        };
        subpasses.push_back(accumPass);

        // Define dependencies
        std::vector<VulkanFramebuffer::SubpassDependency> dependencies =
        {
            {
                VK_SUBPASS_EXTERNAL,
                0,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                0,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_DEPENDENCY_BY_REGION_BIT
            },
            {
                0,
                1,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_DEPENDENCY_BY_REGION_BIT
            }
        };

        uint32_t swapChainWidth = m_Swapchain->GetWidth();
        uint32_t swapChainHeight = m_Swapchain->GetHeight();

        framebuffer = std::make_unique<VulkanFramebuffer>(
                m_DeviceRef,
                swapChainWidth, swapChainHeight,
                attachmentSpecs,
                subpasses,
                dependencies);
    }

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 1.0f;
    samplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(m_DeviceRef.GetDevice(), &samplerCreateInfo, nullptr, &m_FramebufferColorSampler));
}

int GetCompositeAttachmentInputFromFrame(int frameNumber)
{
    // If the frame is even, the accumulated output will be at attachmentC, or the 3rd attachment (index 2).
    // If the frame is odd, the accumulated output will be at attachmentB, or the 2nd attachment (index 1).
    return frameNumber % 2 == 0 ? 2 : 1;
}

void Renderer::SetupMainRayTracePass()
{
    // Layout
    m_MainRTPassDescriptorSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: SSBO for Spheres
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            .Build();

    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_GlobalSetLayout->GetDescriptorSetLayout(),
        m_MainRTPassDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_MainRTPassGraphicsPipelineLayout));

    VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
    VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.RenderPass = m_Framebuffers[0]->GetRenderPass();
    pipelineConfig.PipelineLayout = m_CompositionGraphicsPipelineLayout;
    pipelineConfig.EmptyVertexInputState = true;
    pipelineConfig.Subpass = 0;

    VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState{};
    pipelineColorBlendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
    pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;

    pipelineConfig.ColorBlendInfo.attachmentCount = 1;
    pipelineConfig.ColorBlendInfo.pAttachments = &pipelineColorBlendAttachmentState;

    m_MainRTPassGraphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
            m_DeviceRef,
            "../assets/shaders/raytrace.vert.spv",
            "../assets/shaders/raytrace.frag.spv",
            pipelineConfig);

    m_MainRTPassDescriptorSets.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo info = m_SphereSSBOs[i]->DescriptorInfo();
        VulkanDescriptorWriter(*m_MainRTPassDescriptorSetLayout, *m_OffscreenDescriptorPool)
                .WriteBuffer(0, &info)
                .Build(m_MainRTPassDescriptorSets[i]);
    }
}

void Renderer::SetupAccumulationPass()
{
    // Layout
    m_AccumulationDescriptorSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: Input Attachment - Attachment B or C depending on the frame #
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            .Build();

    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_GlobalSetLayout->GetDescriptorSetLayout(),
        m_MainRTPassDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_AccumulationGraphicsPipelineLayout));

    VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
    VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.RenderPass = m_Framebuffers[0]->GetRenderPass();
    pipelineConfig.PipelineLayout = m_AccumulationGraphicsPipelineLayout;
    pipelineConfig.EmptyVertexInputState = true;
    pipelineConfig.Subpass = 1;

    std::vector<VkPipelineColorBlendAttachmentState> pipelineColorBlendAttachmentStates(2);
    for(auto  pipelineColorBlendAttachmentState : pipelineColorBlendAttachmentStates)
    {
        pipelineColorBlendAttachmentState.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;
    }

    pipelineConfig.ColorBlendInfo.attachmentCount = pipelineColorBlendAttachmentStates.size();
    pipelineConfig.ColorBlendInfo.pAttachments = pipelineColorBlendAttachmentStates.data();

    m_AccumulationPipeline = std::make_unique<VulkanGraphicsPipeline>(
            m_DeviceRef,
            "../assets/shaders/fsq.vert.spv",
            "../assets/shaders/accumulation.frag.spv",
            pipelineConfig);

    m_AccumulationDescriptorSets.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        for(auto& accumDescriptorSet: m_AccumulationDescriptorSets[i])
        {
            // Even frames take attachment 1, odd frames take attachment 2.
            int attachmentIndex = i == 0 ? 1 : 2;
            VkDescriptorImageInfo info = m_Framebuffers[i]->GetDescriptorImageInfoForAttachment(attachmentIndex, m_FramebufferColorSampler);
            VulkanDescriptorWriter(*m_AccumulationDescriptorSetLayout, *m_DescriptorPool)
                    .WriteImage(0, &info)
                    .Build(accumDescriptorSet);
        }
    }
}

void Renderer::ExecuteAccumulationPass(VkCommandBuffer cmdBuffer, int frameNumber)
{

}


void Renderer::SetupCompositionPass()
{
    // Current - A, Previous - B, Accumulated - C
    // Current - A, Previous - C, Accumulated - B
    // Current - A, Previous - B, Accumulated - C

    // Pool
    m_DescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 3)    // 3 Attachments
            .AddPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1)      // Spheres buffer
            .Build();

    // Layout
    m_CompositeDescriptorSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: FBO Attachment Color Sampler from Accumulation Pass
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            .Build();

    m_CompositionDescriptorSets.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    // For each frame in flight, create a descriptor set pair for both cases
    // Case 1: Attachment 1 is the Accumulation pass output
    // Case 2: Attachment 2 is the Accumulation pass output
    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        for(int j = 0; j < 2; j++)
        {
            const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
            {
                m_GlobalSetLayout->GetDescriptorSetLayout(),
                m_CompositeDescriptorSetLayout->GetDescriptorSetLayout()
            };

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
            pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

            VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr, &m_CompositionGraphicsPipelineLayout));

            VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
            VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

            pipelineConfig.RenderPass = m_Swapchain->GetRenderPass();
            pipelineConfig.PipelineLayout = m_CompositionGraphicsPipelineLayout;
            pipelineConfig.EmptyVertexInputState = true;
            m_CompositionGraphicsPipeline = std::make_unique<VulkanGraphicsPipeline>(
                    m_DeviceRef,
                    "../assets/shaders/fsq.vert.spv",
                    "../assets/shaders/texture_display.frag.spv",
                    pipelineConfig);

            int accumulationAttachmentIndex = GetCompositeAttachmentInputFromFrame(j);
            VkDescriptorImageInfo accumulatedAttachment = m_Framebuffers[i]->GetDescriptorImageInfoForAttachment(accumulationAttachmentIndex, m_FramebufferColorSampler);

            VulkanDescriptorWriter(*m_CompositeDescriptorSetLayout, *m_DescriptorPool)
                    .WriteImage(0, &accumulatedAttachment)
                    .Build(m_CompositionDescriptorSets[i][j]);
        }
    }
}

void Renderer::CreateSynchronizationPrimitives()
{
    // Offscreen Sync Primitives
    {
        m_OffscreenRenderFinishedSemaphores.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (size_t i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr, &m_OffscreenRenderFinishedSemaphores[i]));
            std::stringstream semaphoreNameStream;
            semaphoreNameStream << "OffscreenFinished" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_OffscreenRenderFinishedSemaphores[i], semaphoreNameStream.str().c_str());
        }
    }

    // Accumulation Sync Primitives
    {
        m_AccumulationFinishedSemaphores.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        for (size_t i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr, &m_AccumulationFinishedSemaphores[i]));
            std::stringstream semaphoreNameStream;
            semaphoreNameStream << "AccumulationFinished" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_AccumulationFinishedSemaphores[i], semaphoreNameStream.str().c_str());
        }
    }

    // Graphics Sync Primitives
    {
        m_GraphicsSemaphores.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        // These need to be signaled on the first frame
        for (size_t i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr, &m_GraphicsSemaphores[i]));
            std::stringstream semaphoreNameStream;
            semaphoreNameStream << "GraphicsFinished" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_GraphicsSemaphores[i], semaphoreNameStream.str().c_str());
        }
        VkSubmitInfo submitInfo {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.signalSemaphoreCount = m_GraphicsSemaphores.size();
        submitInfo.pSignalSemaphores = m_GraphicsSemaphores.data();

        // Signal these to start so the compute queue isn't waiting forever.
        VK_CHECK_RESULT(vkQueueSubmit(m_DeviceRef.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK_RESULT(vkQueueWaitIdle(m_DeviceRef.GetGraphicsQueue()));
    }

    // Presentation/Render Sync Primitives
    {
        m_PresentCompleteSemaphores.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
        m_RenderCompleteSemaphores.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
        m_WaitFences.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
        m_ImagesInFlightFences.resize(m_Swapchain->GetImageCount(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo = {};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
        {
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr, &m_PresentCompleteSemaphores[i]));
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr, &m_RenderCompleteSemaphores[i]));
            VK_CHECK_RESULT(vkCreateFence(m_DeviceRef.GetDevice(), &fenceInfo, nullptr, &m_WaitFences[i]));
            std::stringstream semaphoreNameStream;
            semaphoreNameStream << "PresentComplete" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_PresentCompleteSemaphores[i], semaphoreNameStream.str().c_str());

            std::stringstream semaphoreRenderNameStream;
            semaphoreRenderNameStream << "RenderComplete" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE, (uint64_t)m_RenderCompleteSemaphores[i], semaphoreRenderNameStream.str().c_str());

            std::stringstream fenceNameStream;
            fenceNameStream << "WaitFence" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_FENCE, (uint64_t)m_WaitFences[i], fenceNameStream.str().c_str());
        }
    }
}

void Renderer::OnSwapchainResized(uint32_t width, uint32_t height)
{
    for(auto& framebuffer : m_Framebuffers)
        framebuffer->Resize(width, height);
}

void Renderer::BuildCommandBuffers(VkDescriptorSet globalSet)
{
    VkCommandBufferBeginInfo cmdBufferBeginInfo {};
    cmdBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VkClearValue clearValues[5];
    clearValues[0].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[1].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[2].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[3].color = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    clearValues[4].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo {};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = m_Framebuffers[0]->GetRenderPass();
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = m_Swapchain->GetWidth();
    renderPassBeginInfo.renderArea.extent.height = m_Swapchain->GetHeight();
    renderPassBeginInfo.clearValueCount = 5;
    renderPassBeginInfo.pClearValues = clearValues;

    for (int32_t i = 0; i < m_DrawCommandBuffers.size(); ++i)
    {
        // Set target frame buffer
        renderPassBeginInfo.framebuffer = m_Framebuffers[i]->GetFramebuffer();

        VK_CHECK_RESULT(vkBeginCommandBuffer(m_DrawCommandBuffers[i], &cmdBufferBeginInfo));

        vkCmdBeginRenderPass(m_DrawCommandBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport {};
        viewport.width = static_cast<float>(m_Swapchain->GetWidth());
        viewport.height = static_cast<float>(m_Swapchain->GetHeight());
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(m_DrawCommandBuffers[i], 0, 1, &viewport);

        VkRect2D scissor {};
        scissor.extent.width = m_Swapchain->GetWidth();
        scissor.extent.height = m_Swapchain->GetHeight();
        scissor.offset.x = 0;
        scissor.offset.y = 0;

        vkCmdSetScissor(m_DrawCommandBuffers[i], 0, 1, &scissor);

        // First subpass - Scene Ray Trace
        // Renders the components of the scene to the active "Current" attachment
        {
            m_OffscreenGraphicsPipeline->Bind(m_DrawCommandBuffers[i]);
            vkCmdBindDescriptorSets(
                    m_DrawCommandBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_OffscreenGraphicsPipelineLayout,
                    0,
                    1,
                    &globalSet,
                    0,
                    nullptr);
            vkCmdDraw(m_DrawCommandBuffers[i], 3, 1, 0, 0);
        }

        // Second subpass - Accumulation
        // current attachment + previous attachment = next
        {
            vkCmdNextSubpass(m_DrawCommandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);

            m_AccumulationPipeline->Bind(m_DrawCommandBuffers[i]);
            vkCmdBindDescriptorSets(
                    m_DrawCommandBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_AccumulationGraphicsPipelineLayout,
                    0,
                    1,
                    &descriptorSets.composition,
                    0,
                    nullptr);
            vkCmdDraw(m_DrawCommandBuffers[i], 3, 1, 0, 0);
        }

        // Third subpass - Composition
        // Writes
        {
            vkCmdNextSubpass(m_DrawCommandBuffers[i], VK_SUBPASS_CONTENTS_INLINE);

            m_CompositionGraphicsPipeline->Bind(m_DrawCommandBuffers[i]);
            vkCmdBindDescriptorSets(
                    m_DrawCommandBuffers[i],
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                     m_CompositionGraphicsPipelineLayout,
                     0,
                     1,
                     &descriptorSets.transparent,
                     0,
                     nullptr);
            vkCmdDraw(m_DrawCommandBuffers[i], 3, 1, 0, 0);
        }

        vkCmdEndRenderPass(m_DrawCommandBuffers[i]);
        VK_CHECK_RESULT(vkEndCommandBuffer(m_DrawCommandBuffers[i]));
    }
}
