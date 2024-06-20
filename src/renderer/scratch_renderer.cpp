#include "scratch_renderer.h"
#include "renderer/vulkan/vulkan_utils.h"
#include "core/frame_info.h"

#include <sstream>

RTRenderer::RTRenderer(Window &windowRef, VulkanDevice &deviceRef)
    :m_WindowRef(windowRef), m_DeviceRef(deviceRef)
{

}

RTRenderer::~RTRenderer()
{

}

void RTRenderer::Initialize()
{
    RecreateSwapchain();
    CreateFramebuffers();
    AllocateCommandBuffers();
    CreateSynchronizationPrimitives();
    SetupGlobalDescriptors();

    SetupMainRayTracePass();
    SetupAccumulationPass();

    for(int i = 0; i < m_DrawCommandBuffers.size(); i++)
    {
        VkDescriptorSet globalSet = m_GlobalDescriptorSets[i];
        VulkanFramebuffer& fbo = *m_Framebuffers[i];

        for(int frameIndex = 0; frameIndex < 2; frameIndex++)
        {
            VkCommandBuffer cmdBuffer = m_DrawCommandBuffers[i][frameIndex];

            // Begin recording the offscreen command buffer
            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
            clearValues[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo renderPassBeginInfo{};
            renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassBeginInfo.renderPass = fbo.GetRenderPass();
            renderPassBeginInfo.framebuffer = fbo.GetFramebuffer();
            renderPassBeginInfo.renderArea.extent.width = fbo.GetWidth();
            renderPassBeginInfo.renderArea.extent.height = fbo.GetHeight();
            renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassBeginInfo.pClearValues = clearValues.data();

            VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

            vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(fbo.GetWidth());
            viewport.height = static_cast<float>(fbo.GetHeight());
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            VkRect2D scissor{{0, 0}, renderPassBeginInfo.renderArea.extent};
            vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
            vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

            {
                VkDescriptorSet mainSet = m_MainRTPassDescriptorSets[i];
                RecordMainRTPass(
                        cmdBuffer,
                        globalSet,
                        mainSet);
            }

            {
                VkDescriptorSet accumulationSet = m_AccumulationDescriptorSets[i][frameIndex];
                VkDescriptorImageInfo curr = fbo.GetDescriptorImageInfoForAttachment(0, m_FramebufferColorSampler);
                // If the frame is even, attachment at index 1 will be the previous frame's output.
                int prevIndex = frameIndex == 0 ? 1 : 2;
                VkDescriptorImageInfo prev = fbo.GetDescriptorImageInfoForAttachment(prevIndex, m_FramebufferColorSampler);
                VulkanDescriptorWriter(*m_AccumulationDescriptorSetLayout, *m_DescriptorPool)
                        .WriteImage(0, &curr)
                        .WriteImage(1, &prev)
                        .Build(accumulationSet);

                VulkanGraphicsPipeline &pipeline = *m_AccumulationPipelines[frameIndex];
                RecordAccumulationPass(
                        cmdBuffer,
                        pipeline,
                        globalSet,
                        accumulationSet);
            }

            vkCmdEndRenderPass(cmdBuffer);
            VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));

            {
                VkFramebuffer swapchainFbo = m_Swapchain->GetFrameBuffer(i);
                // If the frame is even, attachment at index 2 will be the final output.
                int displayAttachmentIndex = frameIndex == 0 ? 2 : 1;
                VkDescriptorImageInfo displayImageInfo = fbo.GetDescriptorImageInfoForAttachment(displayAttachmentIndex,
                                                                                                 m_FramebufferColorSampler);
                VulkanDescriptorWriter(*m_CompositeDescriptorSetLayout, *m_DescriptorPool)
                        .WriteImage(0, &displayImageInfo)
                        .Build(m_CompositionDescriptorSets[i][frameIndex]);

                RecordCompositionPass(
                        cmdBuffer,
                        globalSet,
                        m_CompositionDescriptorSets[i][frameIndex],
                        swapchainFbo);
            }
        }
    }
}

void RTRenderer::RecordMainRTPass(
        VkCommandBuffer cmdBuffer,
        VkDescriptorSet globalSet,
        VkDescriptorSet mainSet)
{
    m_MainRTPassGraphicsPipeline->Bind(cmdBuffer);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_MainRTPassGraphicsPipelineLayout,
            0,
            1,
            &globalSet,
            0,
            nullptr);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_MainRTPassGraphicsPipelineLayout,
            1,
            1,
            &mainSet,
            0,
            nullptr);

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
}

void RTRenderer::RecordAccumulationPass(
        VkCommandBuffer cmdBuffer,
        VulkanGraphicsPipeline &pipeline,
        VkDescriptorSet globalSet,
        VkDescriptorSet accumulationSet)
{
    pipeline.Bind(cmdBuffer);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_AccumulationGraphicsPipelineLayout,
            0,
            1,
            &globalSet,
            0,
            nullptr);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_AccumulationGraphicsPipelineLayout,
            1,
            1,
            &accumulationSet,
            0,
            nullptr);

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
}


void RTRenderer::RecordCompositionPass(
        VkCommandBuffer cmdBuffer,
        VkDescriptorSet globalSet,
        VkDescriptorSet compositionSet,
        VkFramebuffer swapchainFbo)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_Swapchain->GetRenderPass();
    renderPassInfo.framebuffer = swapchainFbo;

    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_Swapchain->GetSwapchainExtent();

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {0.01f, 0.01f, 0.01f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cmdBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(m_Swapchain->GetSwapchainExtent().width);
    viewport.height = static_cast<float>(m_Swapchain->GetSwapchainExtent().height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, m_Swapchain->GetSwapchainExtent()};
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    m_CompositionGraphicsPipeline->Bind(cmdBuffer);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_CompositionGraphicsPipelineLayout,
            0,
            1,
            &globalSet,
            0,
            nullptr);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_CompositionGraphicsPipelineLayout,
            1,
            1,
            &compositionSet,
            0,
            nullptr);

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
}

void RTRenderer::Render(Camera& cameraRef)
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

    VkCommandBuffer cmdBuffer = m_DrawCommandBuffers[m_CurrentFrameIndex][m_AccumulationIndex];

    //Acquisition
    {
        vkWaitForFences(m_DeviceRef.GetDevice(), 1, &m_WaitFences[m_CurrentFrameIndex], VK_TRUE,
                        std::numeric_limits<uint64_t>::max());

        auto result = m_Swapchain->AcquireNextImage(&m_CurrentFrameIndex, m_PresentCompleteSemaphores[m_CurrentFrameIndex]);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            RecreateSwapchain();
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swap chain image!");
    }

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmdBuffer;
    VK_CHECK_RESULT(vkQueueSubmit(m_DeviceRef.GetGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE));

    // Presentation
    {
        auto result = m_Swapchain->Present(
                m_DeviceRef.GetPresentQueue(),
                m_CurrentFrameIndex,
                m_RenderCompleteSemaphores[m_CurrentFrameIndex]);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || m_WindowRef.WasWindowResized())
        {
            m_WindowRef.ResetWindowResizedFlag();
            RecreateSwapchain();
        }
        else if (result != VK_SUCCESS)
        {
            throw std::runtime_error("Failed to present swapchain image!");
        }
        VK_CHECK_RESULT(vkQueueWaitIdle(m_DeviceRef.GetGraphicsQueue()));
    }

    m_FrameCounter++;
    m_AccumulationIndex = (m_AccumulationIndex + 1) % 2;
}

void RTRenderer::AllocateCommandBuffers()
{
    m_DrawCommandBuffers.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandBufferCount = m_DrawCommandBuffers.size();
        allocInfo.commandPool = m_DeviceRef.GetGraphicsCommandPool();
        vkAllocateCommandBuffers(m_DeviceRef.GetDevice(), &allocInfo, m_DrawCommandBuffers[i].data());
    }
}

void RTRenderer::SetupGlobalDescriptors()
{
    m_GlobalSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: Global UBO
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_ALL_GRAPHICS)
            .Build();

    m_DescriptorPool = VulkanDescriptorPool::Builder(m_DeviceRef)
            .SetMaxSets(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 2)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 2)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 2)
            .Build();

    m_GlobalUBOs.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for (auto &uboBuffer: m_GlobalUBOs)
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
    for (int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        auto bufferInfo = m_GlobalUBOs[i]->DescriptorInfo();
        VulkanDescriptorWriter(*m_GlobalSetLayout, *m_DescriptorPool)
                .WriteBuffer(0, &bufferInfo)
                .Build(m_GlobalDescriptorSets[i]);
    }
}

void RTRenderer::CreateFramebuffers()
{
    m_Framebuffers.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for (auto &framebuffer: m_Framebuffers)
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

        VulkanFramebuffer::Subpass accumPassB = {};
        accumPassB.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        // Even frame subpasses - writes to Attachment B (1).
        accumPassB.ColorAttachments =
        {
            {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        };
        subpasses.push_back(accumPassB);

        VulkanFramebuffer::Subpass accumPassC = {};
        accumPassC.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        // Odd frame subpasses - writes to Attachment C (2).
        accumPassC.ColorAttachments =
        {
            {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
        };
        subpasses.push_back(accumPassC);

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
            // From subpass 0 to 1
            {
                    0,
                    1,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_DEPENDENCY_BY_REGION_BIT
            },
            // From subpass 0 to 2
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

void RTRenderer::SetupMainRayTracePass()
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

    VK_CHECK_RESULT(vkCreatePipelineLayout(m_DeviceRef.GetDevice(), &pipelineLayoutInfo, nullptr,
                                           &m_MainRTPassGraphicsPipelineLayout));

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
    for (int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        VkDescriptorBufferInfo info = m_SphereSSBOs[i]->DescriptorInfo();
        VulkanDescriptorWriter(*m_MainRTPassDescriptorSetLayout, *m_DescriptorPool)
                .WriteBuffer(0, &info)
                .Build(m_MainRTPassDescriptorSets[i]);
    }
}

void RTRenderer::SetupAccumulationPass()
{
    // Layout
    m_AccumulationDescriptorSetLayout = VulkanDescriptorSetLayout::Builder(m_DeviceRef)
            // Binding 0: Input Attachment - Attachment B or C depending on the frame #
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            // Binding 1: Input Attachment - Attachment B or C depending on the frame #
            .AddBinding(
                    1,
                    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            .Build();

    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_GlobalSetLayout->GetDescriptorSetLayout(),
        m_AccumulationDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(
            m_DeviceRef.GetDevice(),
            &pipelineLayoutInfo,
            nullptr,

            &m_AccumulationGraphicsPipelineLayout));

    int subpassIndex = 1;
    for(auto& accumPipeline : m_AccumulationPipelines)
    {
        VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
        VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

        pipelineConfig.RenderPass = m_Framebuffers[0]->GetRenderPass();
        pipelineConfig.PipelineLayout = m_AccumulationGraphicsPipelineLayout;
        pipelineConfig.EmptyVertexInputState = true;
        pipelineConfig.Subpass = subpassIndex++;

        VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
        pipelineColorBlendAttachmentState.colorWriteMask =
                VK_COLOR_COMPONENT_R_BIT |
                VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT |
                VK_COLOR_COMPONENT_A_BIT;
        pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;

        pipelineConfig.ColorBlendInfo.attachmentCount = 1;
        pipelineConfig.ColorBlendInfo.pAttachments = &pipelineColorBlendAttachmentState;

        accumPipeline = std::make_unique<VulkanGraphicsPipeline>(
                m_DeviceRef,
                "../assets/shaders/fsq.vert.spv",
                "../assets/shaders/accumulation.frag.spv",
                pipelineConfig);
    }

    m_AccumulationDescriptorSets.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for (int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        for (auto &accumDescriptorSet: m_AccumulationDescriptorSets[i])
        {
            // Even frames take attachment 1, odd frames take attachment 2.
            int attachmentIndex = i == 0 ? 1 : 2;
            VkDescriptorImageInfo info = m_Framebuffers[i]->GetDescriptorImageInfoForAttachment(attachmentIndex,
                                                                                                m_FramebufferColorSampler);
            VulkanDescriptorWriter(*m_AccumulationDescriptorSetLayout, *m_DescriptorPool)
                    .WriteImage(0, &info)
                    .Build(accumDescriptorSet);
        }
    }
}

void RTRenderer::CreateSynchronizationPrimitives()
{
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
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr,
                                              &m_PresentCompleteSemaphores[i]));
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr,
                                              &m_RenderCompleteSemaphores[i]));
            VK_CHECK_RESULT(vkCreateFence(m_DeviceRef.GetDevice(), &fenceInfo, nullptr, &m_WaitFences[i]));
            std::stringstream semaphoreNameStream;
            semaphoreNameStream << "PresentComplete" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE,
                                    (uint64_t) m_PresentCompleteSemaphores[i], semaphoreNameStream.str().c_str());

            std::stringstream semaphoreRenderNameStream;
            semaphoreRenderNameStream << "RenderComplete" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_SEMAPHORE,
                                    (uint64_t) m_RenderCompleteSemaphores[i], semaphoreRenderNameStream.str().c_str());

            std::stringstream fenceNameStream;
            fenceNameStream << "WaitFence" << i;
            SetDebugUtilsObjectName(m_DeviceRef.GetDevice(), VK_OBJECT_TYPE_FENCE, (uint64_t) m_WaitFences[i],
                                    fenceNameStream.str().c_str());
        }
    }
}

void RTRenderer::RecreateSwapchain()
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
    } else
    {
        std::shared_ptr<VulkanSwapchain> oldSwapChain = std::move(m_Swapchain);
        m_Swapchain = std::make_unique<VulkanSwapchain>(m_DeviceRef, extent, oldSwapChain);

        if (!oldSwapChain->CompareSwapchainFormats(*m_Swapchain))
        {
            throw std::runtime_error("Swap chain image(or depth) format has changed!");
        }
    }
}

void RTRenderer::OnSwapchainResized(uint32_t width, uint32_t height)
{
    for (auto &framebuffer: m_Framebuffers)
        framebuffer->Resize(width, height);
}