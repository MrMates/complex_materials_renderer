#include "utils.hpp"

VkCommandBuffer Utils::AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
	VkCommandBufferAllocateInfo cmdAllocInfo = nvvk::make<VkCommandBufferAllocateInfo>();
	cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cmdAllocInfo.commandPool = cmdPool;
	cmdAllocInfo.commandBufferCount = 1;
	VkCommandBuffer cmdBuffer;
	NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
	VkCommandBufferBeginInfo beginInfo = nvvk::make<VkCommandBufferBeginInfo>();
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
	return cmdBuffer;
}

void Utils::EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer)
{
	NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
	VkSubmitInfo submitInfo = nvvk::make<VkSubmitInfo>();
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &cmdBuffer;
	NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
	NVVK_CHECK(vkQueueWaitIdle(queue));
	vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}