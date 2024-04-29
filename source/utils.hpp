#ifndef UTILS_HEADER_FILE
#define UTILS_HEADER_FILE

#include <charconv>
#include <iostream>

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/context_vk.hpp>
#include <nvvk/error_vk.hpp>

#include <tiny_obj_loader.h>

class Options
{
public:
	int numSamples = 256;
	int backgroundTexture = 1;
	std::string_view objPath = "resources/scenes/studio_corner.obj";
	std::string_view outName = "out";
	float cameraPos[3] = { 0.2f, 4.2f, 6.5f };
	float cameraLookAt[3] = { 0.0f, 4.1f, 0.2f };
	float cameraFOV = 36.0;
	float lightPos[3] = {-1.001f, 5.0f, 6.0f};
	float lightColor[3] = {0.8f, 0.8f, 0.6f};
	float lightIntensity = 100.0f;
	float scale = 10.0f;
};

static class Utils
{
public:
	static VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool);
	static void EndSubmitWaitAndFreeCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool cmdPool, VkCommandBuffer& cmdBuffer);

	static void parse(int argc, char* argv[], Options* options);
};

#endif // END UTILS_HEADER_FILE