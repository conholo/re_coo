#pragma once

#include "renderer/vulkan/vulkan_framebuffer.h"
#include "renderer/vulkan/vulkan_swapchain.h"
#include "renderer/vulkan/vulkan_descriptors.h"
#include "renderer/vulkan/vulkan_graphics_pipeline.h"
#include "renderer/vulkan/vulkan_buffer.h"
#include "renderer/camera.h"
#include <memory>
#include <vector>
#include <vulkan/vulkan.h>

class RTRenderer
{
public:
    explicit RTRenderer(Window& windowRef, VulkanDevice &deviceRef);
    ~RTRenderer();

    void Initialize();
    void Render(Camera &cameraRef);

    float GetAspectRatio() const { return m_Swapchain->GetExtentAspectRatio(); }

private:

    void RecordMainRTPass(
            VkCommandBuffer cmdBuffer,
            VkDescriptorSet globalSet,
            VkDescriptorSet mainSet);

    void RecordAccumulationPass(
            VkCommandBuffer cmdBuffer,
            VulkanGraphicsPipeline &pipeline,
            VkDescriptorSet globalSet,
            VkDescriptorSet accumulationSet);

    void RecordCompositionPass(
            VkCommandBuffer cmdBuffer,
            VkDescriptorSet globalSet,
            VkDescriptorSet compositionSet,
            VkFramebuffer swapchainFbo);

    void CreateFramebuffers();
    void AllocateCommandBuffers();
    void SetupGlobalDescriptors();
    void CreateSynchronizationPrimitives();

    void SetupMainRayTracePass();
    void SetupAccumulationPass();

    void RecreateSwapchain();
    void OnSwapchainResized(uint32_t width, uint32_t height);


private:
    Window& m_WindowRef;
    VulkanDevice& m_DeviceRef;
    std::unique_ptr<VulkanSwapchain> m_Swapchain;

    std::vector<std::array<VkCommandBuffer, 2>> m_DrawCommandBuffers;

    std::unique_ptr<VulkanDescriptorPool> m_DescriptorPool;
    std::vector<std::unique_ptr<VulkanFramebuffer>> m_Framebuffers;
    VkSampler m_FramebufferColorSampler;

    // UBOs
    std::vector<std::unique_ptr<VulkanBuffer>> m_GlobalUBOs;

    // Buffers
    std::vector<std::unique_ptr<VulkanBuffer>> m_SphereSSBOs;

    // Descriptor Set Layouts
    std::unique_ptr<VulkanDescriptorSetLayout> m_MainRTPassDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_AccumulationDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_CompositeDescriptorSetLayout;
    std::unique_ptr<VulkanDescriptorSetLayout> m_GlobalSetLayout;

    // Descriptor Sets
    std::vector<std::array<VkDescriptorSet, 2>> m_CompositionDescriptorSets;
    std::vector<std::array<VkDescriptorSet, 2>> m_AccumulationDescriptorSets;
    std::vector<VkDescriptorSet> m_MainRTPassDescriptorSets;
    std::vector<VkDescriptorSet> m_GlobalDescriptorSets;

    // Pipeline Layouts
    VkPipelineLayout m_MainRTPassGraphicsPipelineLayout{};
    VkPipelineLayout m_AccumulationGraphicsPipelineLayout{};
    VkPipelineLayout m_CompositionGraphicsPipelineLayout{};

    // Pipelines
    std::unique_ptr<VulkanGraphicsPipeline> m_CompositionGraphicsPipeline;
    std::unique_ptr<VulkanGraphicsPipeline> m_MainRTPassGraphicsPipeline;
    std::array<std::unique_ptr<VulkanGraphicsPipeline>, 2> m_AccumulationPipelines;

    std::vector<VkSemaphore> m_PresentCompleteSemaphores;   // Swap chain image presentation
    std::vector<VkSemaphore> m_RenderCompleteSemaphores;    // Command buffer submission and execution

    std::vector<VkFence> m_WaitFences;
    std::vector<VkFence> m_ImagesInFlightFences;

    VkSemaphore m_RenderComplete;

    uint64_t m_FrameCounter = 0;
    uint32_t m_CurrentFrameIndex = 0;
    uint8_t m_AccumulationIndex = 0;
};