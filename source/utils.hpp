#ifndef UTILS_HEADER_FILE
#define UTILS_HEADER_FILE

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/error_vk.hpp>

#include <tiny_obj_loader.h>

static class Utils
{
public:
	static VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool);
	static void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer);
};

#endif // END UTILS_HEADER_FILE