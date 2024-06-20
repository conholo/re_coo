#include "scratch_renderer.h"
#include "renderer/vulkan/vulkan_utils.h"
#include "core/frame_info.h"
#include "scene/scene.h"

#include <sstream>

RTRenderer::RTRenderer(Window &windowRef, VulkanDevice &deviceRef)
    :m_WindowRef(windowRef), m_DeviceRef(deviceRef)
{

}

RTRenderer::~RTRenderer()
{

}

void RTRenderer::CreateSphereBuffers()
{
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

void RTRenderer::RecordFrame(int swapImageIndex, VkDescriptorSet globalSet)
{
    // If the frame is even, the previous fbo is at index 1.
    int prevIndex = m_FrameCounter % 2 == 0 ? 1 : 0;
    int currIndex = m_FrameCounter % 2 == 0 ? 0 : 1;
    VulkanFramebuffer& prevFbo = *m_PerFrameFramebufferMap[swapImageIndex][prevIndex];
    VulkanFramebuffer& currFbo = *m_PerFrameFramebufferMap[swapImageIndex][currIndex];
    VkCommandBuffer cmdBuffer = m_DrawCommandBuffers[swapImageIndex];

    // Begin recording the offscreen command buffer
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = currFbo.GetRenderPass();
    renderPassBeginInfo.framebuffer = currFbo.GetFramebuffer();
    renderPassBeginInfo.renderArea.extent.width = currFbo.GetWidth();
    renderPassBeginInfo.renderArea.extent.height = currFbo.GetHeight();
    renderPassBeginInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
    renderPassBeginInfo.pClearValues = clearValues.data();

    VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuffer, &beginInfo));

    vkCmdBeginRenderPass(cmdBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(currFbo.GetWidth());
    viewport.height = static_cast<float>(currFbo.GetHeight());
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    VkRect2D scissor{{0, 0}, renderPassBeginInfo.renderArea.extent};
    vkCmdSetViewport(cmdBuffer, 0, 1, &viewport);
    vkCmdSetScissor(cmdBuffer, 0, 1, &scissor);

    {
        VkDescriptorSet mainSet = m_MainRTPassDescriptorSets[swapImageIndex];
        RecordMainRTPass(
                cmdBuffer,
                globalSet,
                mainSet);
    }

    {
        vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);

        VkDescriptorSet accumulationSet;
        auto bufferInfo = m_GlobalUBOs[swapImageIndex]->DescriptorInfo();
        VkDescriptorImageInfo curr = currFbo.GetDescriptorImageInfoForAttachment(0, m_FramebufferColorSampler);
        VkDescriptorImageInfo prev = prevFbo.GetDescriptorImageInfoForAttachment(1, m_FramebufferColorSampler);
        VulkanDescriptorWriter(*m_AccumulationDescriptorSetLayout, *m_DescriptorPool)
                .WriteBuffer(0, &bufferInfo)
                .WriteImage(1, &curr)
                .WriteImage(2, &prev)
                .Build(accumulationSet);

        RecordAccumulationPass(
                cmdBuffer,
                globalSet,
                accumulationSet);
    }

    {
        vkCmdNextSubpass(cmdBuffer, VK_SUBPASS_CONTENTS_INLINE);

        VkDescriptorSet compositionSet;
        VkDescriptorImageInfo accumulatedAttachment = currFbo.GetDescriptorImageInfoForAttachment(1, m_FramebufferColorSampler);
        VulkanDescriptorWriter(*m_CompositeDescriptorSetLayout, *m_DescriptorPool)
                .WriteImage(0, &accumulatedAttachment)
                .Build(compositionSet);

        RecordCompositionPass(
                cmdBuffer,
                compositionSet,
                currFbo);
    }

    vkCmdEndRenderPass(cmdBuffer);
    VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuffer));
}

void RTRenderer::Initialize()
{
    CreateSphereBuffers();
    RecreateSwapchain();
    CreateFramebuffers();
    AllocateCommandBuffers();
    CreateSynchronizationPrimitives();
    SetupGlobalDescriptors();

    SetupMainRayTracePass();
    SetupAccumulationPass();
    SetupCompositionPass();

    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; i++)
    {
        RecordFrame(i, m_GlobalDescriptorSets[i]);
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
        VkDescriptorSet globalSet,
        VkDescriptorSet accumulationSet)
{
    m_AccumulationPipeline->Bind(cmdBuffer);

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
        VkDescriptorSet compositionSet,
        VulkanFramebuffer& fbo)
{
    m_CompositionGraphicsPipeline->Bind(cmdBuffer);

    vkCmdBindDescriptorSets(
            cmdBuffer,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            m_CompositionGraphicsPipelineLayout,
            0,
            1,
            &compositionSet,
            0,
            nullptr);

    vkCmdDraw(cmdBuffer, 3, 1, 0, 0);
}

void RTRenderer::Draw(Camera& cameraRef)
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

    //Acquisition
    {
        auto result = m_Swapchain->AcquireNextImage(&m_CurrentFrameIndex, m_PresentCompleteSemaphores[m_CurrentFrameIndex]);

        if (result == VK_ERROR_OUT_OF_DATE_KHR)
        {
            RecreateSwapchain();
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
            throw std::runtime_error("Failed to acquire swap chain image!");
    }

    VkCommandBuffer cmdBuffer = m_DrawCommandBuffers[m_CurrentFrameIndex];

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
            OnSwapchainResized(m_Swapchain->GetWidth(), m_Swapchain->GetHeight());
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
        vkAllocateCommandBuffers(m_DeviceRef.GetDevice(), &allocInfo, m_DrawCommandBuffers.data());
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
            .SetMaxSets(8)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 10)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 10)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 10)
            .AddPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VulkanSwapchain::MAX_FRAMES_IN_FLIGHT * 10)
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

std::unique_ptr<VulkanFramebuffer> CreateFramebuffer(int frameIndex, VulkanDevice& device, VulkanSwapchain& swapchain)
{
    VulkanFramebuffer::Attachment::Specification attachment0 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
        };
    VulkanFramebuffer::Attachment::Specification attachment1 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
        };
    VulkanFramebuffer::Attachment::Specification attachment2 =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
        };
    VulkanFramebuffer::Attachment::Specification swapchainImageAttachment =
        {
            VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT
        };

    std::vector<VulkanFramebuffer::Attachment::Specification> attachmentSpecs =
        {
            attachment0,                // Attachment A
            attachment1,                // Attachment B
            attachment2,                // Attachment C
            swapchainImageAttachment    // Swapchain Image Attachment
        };

    // Define subpasses
    std::vector<VulkanFramebuffer::Subpass> subpasses;

    VulkanFramebuffer::Subpass rayTraceSubpass = {};
    rayTraceSubpass.DepthStencilAttachment.attachment = VK_ATTACHMENT_UNUSED;
    rayTraceSubpass.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    rayTraceSubpass.ColorAttachments =
    {
        {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}
    };
    subpasses.push_back(rayTraceSubpass);

    VulkanFramebuffer::Subpass accumPass = {};
    accumPass.DepthStencilAttachment.attachment = VK_ATTACHMENT_UNUSED;
    accumPass.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    // If it's an even pass, we read from attachment 1 from the previous frame's framebuffer.
    // If it's an odd pass, we read from attachment 2 from the previous frame's framebuffer.
    // However, regardless of parity, we read from attachment 0 from the current frame's framebuffer.
    uint32_t readAttachmentIndex = frameIndex % 2 == 0 ? 1 : 2;
    accumPass.InputAttachments =
    {
        { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
        { readAttachmentIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }
    };

    // If it's an even pass, we write to attachment 2 of the current frame's framebuffer.
    // If it's an odd pass, we write to attachment 1 from the current frame's framebuffer.
    uint32_t writeAttachmentIndex = frameIndex % 2 == 0 ? 2 : 1;
    accumPass.ColorAttachments =
    {
        {writeAttachmentIndex, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
    };
    subpasses.push_back(accumPass);

    VulkanFramebuffer::Subpass finalPass = {};
    accumPass.DepthStencilAttachment.attachment = swapchain.GetImageView();
    accumPass.BindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    // Writes to swapchain image
    accumPass.ColorAttachments =
    {
        {3, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
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
        },
    };

    return std::make_unique<VulkanFramebuffer>(
            device,
            swapchain.GetWidth(), swapchain.GetHeight(),
            attachmentSpecs,
            subpasses,
            dependencies);
}

void RTRenderer::CreateFramebuffers()
{
    m_PerFrameFramebufferMap.resize(VulkanSwapchain::MAX_FRAMES_IN_FLIGHT);
    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        std::array<std::unique_ptr<VulkanFramebuffer>, 2> fbos;
        fbos[0] = std::move(CreateFramebuffer(m_DeviceRef, *m_Swapchain));
        fbos[1] = std::move(CreateFramebuffer(m_DeviceRef, *m_Swapchain));
        m_PerFrameFramebufferMap[i] = std::move(fbos);
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
            // Binding 0: SS BO for Spheres
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

    pipelineConfig.RenderPass = m_PerFrameFramebufferMap[0][0]->GetRenderPass();
    pipelineConfig.PipelineLayout = m_MainRTPassGraphicsPipelineLayout;
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
            // Binding 0: Global UBO
            .AddBinding(
                    0,
                    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            // Binding 1: Input Attachment - Attachment B or C depending on the frame #
            .AddBinding(
                    1,
                    VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,
                    VK_SHADER_STAGE_FRAGMENT_BIT)
            // Binding 2: Input Attachment - Attachment B or C depending on the frame #
            .AddBinding(
                    2,
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

    VulkanGraphicsPipeline::PipelineConfigInfo pipelineConfig{};
    VulkanGraphicsPipeline::GetDefaultPipelineConfigInfo(pipelineConfig);

    pipelineConfig.RenderPass = m_PerFrameFramebufferMap[0][0]->GetRenderPass();
    pipelineConfig.PipelineLayout = m_AccumulationGraphicsPipelineLayout;
    pipelineConfig.EmptyVertexInputState = true;
    pipelineConfig.Subpass = 1;

    VkPipelineColorBlendAttachmentState pipelineColorBlendAttachmentState;
    pipelineColorBlendAttachmentState.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
    pipelineColorBlendAttachmentState.blendEnable = VK_FALSE;

    pipelineConfig.ColorBlendInfo.attachmentCount = 1;
    pipelineConfig.ColorBlendInfo.pAttachments = &pipelineColorBlendAttachmentState;

    m_AccumulationPipeline = std::make_unique<VulkanGraphicsPipeline>(
            m_DeviceRef,
            "../assets/shaders/fsq.vert.spv",
            "../assets/shaders/accumulation.frag.spv",
            pipelineConfig);
}

void RTRenderer::SetupCompositionPass()
{
    m_CompositeDescriptorSetLayout =
            VulkanDescriptorSetLayout::Builder(m_DeviceRef)
                    // Binding 0: FBO Attachment Color Sampler from Accumulation Pass
                    .AddBinding(
                            0,
                            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                            VK_SHADER_STAGE_FRAGMENT_BIT)
                    .Build();

    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts
    {
        m_CompositeDescriptorSetLayout->GetDescriptorSetLayout()
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayouts.size());
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK_RESULT(vkCreatePipelineLayout(
            m_DeviceRef.GetDevice(),
            &pipelineLayoutInfo,
            nullptr,
            &m_CompositionGraphicsPipelineLayout));

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

void RTRenderer::CreateSynchronizationPrimitives()
{
    // Presentation/Draw Sync Primitives
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
            VK_CHECK_RESULT(vkCreateSemaphore(m_DeviceRef.GetDevice(), &semaphoreInfo, nullptr,&m_RenderCompleteSemaphores[i]));
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
    for(int i = 0; i < VulkanSwapchain::MAX_FRAMES_IN_FLIGHT; ++i)
    {
        for (auto &framebuffer: m_PerFrameFramebufferMap[i])
            framebuffer->Resize(width, height);
    }
}