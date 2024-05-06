/*
* File: model.hpp
* Author: Matìj Toul (xtoulm00)
* E-mail: matej.toul@gmail.com
* Date: 6 May 2024
*/

#ifndef MODEL_HEADER_FILE
#define MODEL_HEADER_FILE

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/context_vk.hpp>

#include <tiny_obj_loader.h>
#include <utils.hpp>

class Model
{
public:
	Model(tinyobj::ObjReader reader,
		nvvk::ResourceAllocatorDedicated& allocator,
		nvvk::Context& context,
		VkCommandPool& cmdPool,
		std::string filepath,
		Options* options);

	nvvk::RaytracingBuilderKHR::BlasInput GetBLASInput(nvvk::Context& context);

	nvvk::Buffer vertexBuffer, indexBuffer, materialIdBuffer, mediaDefinitionsBuffer;
private:
	VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);
	std::vector<tinyobj::real_t> objVertices;
	std::vector<uint32_t> objIndices;
	std::vector<int> objMaterials;
	std::vector<float> mediaDefinitions;
};

#endif // END MODEL_HEADER_FILE