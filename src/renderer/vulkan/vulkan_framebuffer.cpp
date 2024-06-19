#include <array>
#include "vulkan_framebuffer.h"
#include "vulkan_utils.h"

VulkanFramebuffer::VulkanFramebuffer(
        VulkanDevice& deviceRef,
        uint32_t width, uint32_t height,
        const std::vector<Attachment::Specification>& attachmentSpecs,
        const std::vector<Subpass>& subpasses,
        const std::vector<SubpassDependency>& dependencies)
        : m_DeviceRef(deviceRef), m_Width(width), m_Height(height), m_Subpasses(subpasses), m_Dependencies(dependencies)
{
    m_Attachments.resize(attachmentSpecs.size());
    for (int i = 0; i < attachmentSpecs.size(); i++)
    {
        CreateAttachment(
                m_DeviceRef,
                attachmentSpecs[i],
                &m_Attachments[i]);
    }

    CreateFramebuffer();
}

VulkanFramebuffer::~VulkanFramebuffer()
{
    for(auto& attachment : m_Attachments)
    {
        vkDestroyImageView(m_DeviceRef.GetDevice(), attachment.View, nullptr);
        vkDestroyImage(m_DeviceRef.GetDevice(), attachment.Image, nullptr);
        vkFreeMemory(m_DeviceRef.GetDevice(), attachment.Mem, nullptr);
    }

    vkDestroyFramebuffer(m_DeviceRef.GetDevice(), m_Framebuffer, nullptr);
    vkDestroyRenderPass(m_DeviceRef.GetDevice(), m_RenderPass, nullptr);
}

void PrintAttachment(VulkanFramebuffer::Attachment& attachment)
{
    std::cout << "Attachment Specifications:\n"
              << "\tVkFormat: " << attachment.Spec.Format << "\n"
              << "\tVkImageUsageFlagBits: " << attachment.Spec.Usage << "\n"
              << "VkImage: " << attachment.Image << "\n"
              << "VkDeviceMemory: " << attachment.Mem << "\n"
              << "VkImageView: " << attachment.View << "\n";
}

void VulkanFramebuffer::CreateAttachment(
    VulkanDevice& deviceRef,
    const Attachment::Specification& spec,
    Attachment* attachment) const
{
    VkImageAspectFlags aspectMask = 0;

    if (spec.Usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }
    if (spec.Usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
        aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        if (spec.Format >= VK_FORMAT_D16_UNORM_S8_UINT)
            aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
    }

    assert(aspectMask > 0);

    attachment->Spec = spec;
    VkImageCreateInfo imageCreateInfo{};
    imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imageCreateInfo.format = spec.Format;
    imageCreateInfo.extent.width = m_Width;
    imageCreateInfo.extent.height = m_Height;
    imageCreateInfo.extent.depth = 1;
    imageCreateInfo.mipLevels = 1;
    imageCreateInfo.arrayLayers = 1;
    imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageCreateInfo.usage = spec.Usage | VK_IMAGE_USAGE_SAMPLED_BIT;

    deviceRef.CreateImageWithInfo(
        imageCreateInfo,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        attachment->Image,
        attachment->Mem);

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = spec.Format;
    viewInfo.subresourceRange = {};
    viewInfo.subresourceRange.aspectMask = aspectMask;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.image = attachment->Image;
    VK_CHECK_RESULT(vkCreateImageView(deviceRef.GetDevice(), &viewInfo, nullptr, &attachment->View));

    PrintAttachment(*attachment);
}

VkDescriptorImageInfo VulkanFramebuffer::GetDescriptorImageInfoForAttachment(uint32_t attachmentIndex, VkSampler sampler) const
{
    assert(attachmentIndex < m_Attachments.size() && "Invalid attachment index.");
    VkDescriptorImageInfo descriptorImageInfo{};
    descriptorImageInfo.sampler = sampler;
    descriptorImageInfo.imageView = m_Attachments[attachmentIndex].View;
    descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return descriptorImageInfo;
}

void VulkanFramebuffer::CreateFramebuffer()
{
    std::vector<VkAttachmentDescription> attachmentDescs{m_Attachments.size()};
    std::vector<VkImageView> attachmentImageViews;

    for (uint32_t i = 0; i < m_Attachments.size(); ++i)
    {
        attachmentDescs[i].samples = VK_SAMPLE_COUNT_1_BIT;
        attachmentDescs[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachmentDescs[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachmentDescs[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachmentDescs[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachmentDescs[i].format = m_Attachments[i].Spec.Format;

        if (m_Attachments[i].Spec.Usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        {
            attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        }
        else
        {
            attachmentDescs[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachmentDescs[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        attachmentImageViews.push_back(m_Attachments[i].View);
    }

    std::vector<VkSubpassDescription> subpassDescs(m_Subpasses.size());
    for (size_t i = 0; i < m_Subpasses.size(); ++i)
    {
        subpassDescs[i].pipelineBindPoint = m_Subpasses[i].BindPoint;
        subpassDescs[i].colorAttachmentCount = static_cast<uint32_t>(m_Subpasses[i].ColorAttachments.size());
        subpassDescs[i].pColorAttachments = m_Subpasses[i].ColorAttachments.data();
        subpassDescs[i].pDepthStencilAttachment =
                m_Subpasses[i].DepthStencilAttachment.attachment != VK_ATTACHMENT_UNUSED
                ? &m_Subpasses[i].DepthStencilAttachment
                : nullptr;
    }

    std::vector<VkSubpassDependency> dependencies(m_Dependencies.size());
    for (size_t i = 0; i < m_Dependencies.size(); ++i)
    {
        dependencies[i] = {
                m_Dependencies[i].SrcSubpass,
                m_Dependencies[i].DstSubpass,
                m_Dependencies[i].SrcStageMask,
                m_Dependencies[i].DstStageMask,
                m_Dependencies[i].SrcAccessMask,
                m_Dependencies[i].DstAccessMask,
                m_Dependencies[i].DependencyFlags
        };
    }

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescs.size());
    renderPassInfo.pAttachments = attachmentDescs.data();
    renderPassInfo.subpassCount = static_cast<uint32_t>(subpassDescs.size());
    renderPassInfo.pSubpasses = subpassDescs.data();
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    VK_CHECK_RESULT(vkCreateRenderPass(m_DeviceRef.GetDevice(), &renderPassInfo, nullptr, &m_RenderPass));

    VkFramebufferCreateInfo fbufCreateInfo = {};
    fbufCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbufCreateInfo.pNext = nullptr;
    fbufCreateInfo.renderPass = m_RenderPass;
    fbufCreateInfo.pAttachments = attachmentImageViews.data();
    fbufCreateInfo.attachmentCount = static_cast<uint32_t>(attachmentImageViews.size());
    fbufCreateInfo.width = m_Width;
    fbufCreateInfo.height = m_Height;
    fbufCreateInfo.layers = 1;
    VK_CHECK_RESULT(vkCreateFramebuffer(m_DeviceRef.GetDevice(), &fbufCreateInfo, nullptr, &m_Framebuffer));
}

void VulkanFramebuffer::Resize(uint32_t width, uint32_t height)
{
    if (width == m_Width && height == m_Height)
        return; // Size unchanged, no need to resize

    m_Width = width;
    m_Height = height;

    // Destroy existing framebuffer
    vkDestroyFramebuffer(m_DeviceRef.GetDevice(), m_Framebuffer, nullptr);
    // Destroy existing render pass
    vkDestroyRenderPass(m_DeviceRef.GetDevice(), m_RenderPass, nullptr);

    // Destroy attachments
    for (auto& attachment : m_Attachments)
    {
        vkDestroyImageView(m_DeviceRef.GetDevice(), attachment.View, nullptr);
        vkDestroyImage(m_DeviceRef.GetDevice(), attachment.Image, nullptr);
        vkFreeMemory(m_DeviceRef.GetDevice(), attachment.Mem, nullptr);
    }

    // Recreate attachments with new size
    for (auto& attachment : m_Attachments)
        CreateAttachment(m_DeviceRef, attachment.Spec, &attachment);

    CreateFramebuffer();
}