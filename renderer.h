#pragma once

#include <memory>
#include "core/frame_info.h"

#include "renderer/vulkan/vulkan_descriptors.h"
#include "renderer/vulkan/vulkan_framebuffer.h"
#include "renderer/vulkan/vulkan_device.h"
#include "renderer/vulkan/vulkan_graphics_pipeline.h"
#include "renderer/vulkan/vulkan_framebuffer.h"
#include "renderer/vulkan/vulkan_swapchain.h"
#include "renderer/vulkan/vulkan_buffer.h"

class Renderer
{
public:
    explicit Renderer(Window& windowRef, VulkanDevice& deviceRef);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;
    void Render(Camera& cameraRef);

    [[nodiscard]] float GetAspectRatio() const { return m_Swapchain->GetExtentAspectRatio(); }

private:

    void BeginOffscreenRenderPass(VkCommandBuffer offscreenCommandBuffer, VulkanBuffer& buffer);
    void RenderSceneOffscreen(VkCommandBuffer offscreenCommandBuffer, VkDescriptorSet globalSet, VulkanBuffer& buffer);
    void EndOffscreenRenderPass(VkCommandBuffer offscreenCommandBuffer);

    void BeginSwapchainRenderPass(VkCommandBuffer drawCommandBuffer, uint32_t bufferIndex);
    void RenderComposition(VkCommandBuffer commandBuffer, VkDescriptorSet globalSet);
    void EndSwapchainRenderPass(VkCommandBuffer drawCommandBuffer);

    // Resource Setup
    void CreateFramebuffers();
    void CreateSynchronizationPrimitives();

    void SetupGlobalDescriptors();
    void SetupCompositeDescriptors();
    void SetupOffscreenDescriptors();
    void SetupAccumulationDescriptors();

    void CreateOffscreenGraphicsPipeline();
    void CreateOffscreenGraphicsPipelineLayout();

    void CreateAccumulationGraphicsPipeline();
    void CreateAccumulationGraphicsPipelineLayout();

    void CreateCompositionGraphicsPipeline();
    void CreateCompositionGraphicsPipelineLayout();

    void RecreateSwapchain();
    void OnSwapchainResized(uint32_t width, uint32_t height);

private:
    Window& m_WindowRef;
    VulkanDevice& m_DeviceRef;
    std::unique_ptr<VulkanSwapchain> m_Swapchain;
    QueueFamilyIndices m_QueueFamily;

    uint32_t m_CurrentBufferIndex = 0;
    uint32_t m_CurrentFrameIndex = 0;
    uint64_t m_FrameCounter = 0;

    // Buffers
    std::vector<std::unique_ptr<VulkanBuffer>> m_SphereSSBOs;

    // UBOs
    std::vector<std::unique_ptr<VulkanBuffer>> m_GlobalUBOs;

    // Framebuffer
    std::vector<std::unique_ptr<VulkanFramebuffer>> m_Framebuffers;
    VkSampler m_FramebufferColorSampler;

    // Command Buffers
    std::vector<VkCommandBuffer> m_DrawCommandBuffers;

    // Descriptor Pools
    std::unique_ptr<VulkanDescriptorPool> m_OffscreenDescriptorPool;
    std::unique_ptr<VulkanDescriptorPool> m_AccumulationDescriptorPool;
    std::unique_ptr<VulkanDescriptorPool> m_DescriptorPool;
    std::unique_ptr<VulkanDescriptorPool> m_GlobalDescriptorPool;

    // Descriptor Set Layouts
    std::unique_ptr<VulkanDescriptorSetLayout> m_MainRTPassDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_AccumulationDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_CompositeDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_GlobalSetLayout;

    // Descriptor Sets
    std::vector<std::array<VkDescriptorSet, 2>> m_CompositionDescriptorSets;
    std::vector<std::array<VkDescriptorSet, 2>> m_AccumulationDescriptorSets;
    std::vector<VkDescriptorSet> m_MainRTPassDescriptorSets;

    // Pipeline Layouts
    VkPipelineLayout m_MainRTPassGraphicsPipelineLayout{};
    VkPipelineLayout m_AccumulationGraphicsPipelineLayout{};
    VkPipelineLayout m_CompositionGraphicsPipelineLayout{};

    // Pipelines
    std::unique_ptr<VulkanGraphicsPipeline> m_CompositionGraphicsPipeline;
    std::unique_ptr<VulkanGraphicsPipeline> m_MainRTPassGraphicsPipeline;
    std::unique_ptr<VulkanGraphicsPipeline> m_AccumulationPipeline;

    // Fences
    std::vector<VkFence> m_OffscreenInFlightFences;
    std::vector<VkFence> m_WaitFences;
    std::vector<VkFence> m_ImagesInFlightFences;

    // Semaphores
    std::vector<VkSemaphore> m_OffscreenRenderFinishedSemaphores;
    std::vector<VkSemaphore> m_AccumulationFinishedSemaphores;
    std::vector<VkSemaphore> m_GraphicsSemaphores;
    std::vector<VkSemaphore> m_PresentCompleteSemaphores;   // Swap chain image presentation
    std::vector<VkSemaphore> m_RenderCompleteSemaphores;    // Command buffer submission and execution
};