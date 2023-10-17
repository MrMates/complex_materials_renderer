#pragma once

#include <nvvk/raytraceKHR_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/context_vk.hpp>

#include <tiny_obj_loader.h>

class Model
{
public:
	Model(tinyobj::ObjReader reader);

	nvvk::RaytracingBuilderKHR::BlasInput GetBLASInput(
		nvvk::Context& context, nvvk::Buffer& vertexBuffer, nvvk::Buffer& indexBuffer);

	std::vector<tinyobj::real_t> objVertices;
	std::vector<uint32_t> objIndices;
private:
	VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer);


};