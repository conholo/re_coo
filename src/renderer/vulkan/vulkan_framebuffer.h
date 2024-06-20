#pragma once

#include "renderer/vulkan/vulkan_device.h"
#include <glm/fwd.hpp>
#include <vulkan/vulkan.h>

class VulkanFramebuffer
{
public:
    struct Attachment
    {
        struct Specification
        {
            VkFormat Format;
            VkImageUsageFlags Usage;
        };

        VkImage Image;
        VkDeviceMemory Mem;
        VkImageView View;
        Specification Spec;
    };

    struct Subpass
    {
        std::vector<VkAttachmentReference> ColorAttachments;
        VkAttachmentReference DepthStencilAttachment;
        std::vector<VkAttachmentReference> InputAttachments;
        VkPipelineBindPoint BindPoint;
    };

    struct SubpassDependency
    {
        uint32_t SrcSubpass;
        uint32_t DstSubpass;
        VkPipelineStageFlags SrcStageMask;
        VkPipelineStageFlags DstStageMask;
        VkAccessFlags SrcAccessMask;
        VkAccessFlags DstAccessMask;
        VkDependencyFlags DependencyFlags;
    };

    VulkanFramebuffer(
            VulkanDevice& deviceRef,
            uint32_t width, uint32_t height,
            const std::vector<Attachment::Specification>& attachmentSpecs,
            const std::vector<Subpass>& subpasses,
            const std::vector<SubpassDependency>& dependencies);
    ~VulkanFramebuffer();

    void CreateAttachment(
        VulkanDevice& deviceRef,
        const Attachment::Specification& spec,
        Attachment* attachment) const;

    void Resize(uint32_t width, uint32_t height);

    [[nodiscard]] VkRenderPass GetRenderPass() const { return m_RenderPass; }
    [[nodiscard]] VkFramebuffer GetFramebuffer() const { return m_Framebuffer; }
    [[nodiscard]] uint32_t GetWidth() const { return m_Width; }
    [[nodiscard]] uint32_t GetHeight() const { return m_Height; }

    VkDescriptorImageInfo GetDescriptorImageInfoForAttachment(uint32_t attachmentIndex, VkSampler sampler) const;

private:
    void CreateFramebuffer();

private:
    VulkanDevice& m_DeviceRef;

    VkFramebuffer m_Framebuffer{};
    VkRenderPass m_RenderPass{};
    uint32_t m_Width, m_Height;
    std::vector<Attachment> m_Attachments;
    std::vector<Subpass> m_Subpasses;
    std::vector<SubpassDependency> m_Dependencies;
};
