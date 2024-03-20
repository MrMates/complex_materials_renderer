#include <cassert>
#include <array>
#include <iostream>
#include <random>

#include <nvvk/context_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>
#include <nvh/fileoperations.hpp>

#include "model.hpp"
#include "utils.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <GLFW/glfw3.h>

static const uint64_t render_width = 1920;
static const uint64_t render_height = 1080;

static const uint64_t workgroup_width = 32;
static const uint64_t workgroup_height = 32;


int main(int argc, const char** argv)
{
	/* DEVICE SETUP */

	// Create the Vulkan context, consisting of an instance, device, physical device, and queues.
	nvvk::ContextCreateInfo deviceInfo;  // One can modify this to load different extensions or pick the Vulkan core version
	nvvk::Context           context;     // Encapsulates device state in a single object

	deviceInfo.apiMajor = 1;
	deviceInfo.apiMinor = 3;

	// Required by VK_KHR_ray_query; allows work to be offloaded onto background threads and parallelized
	deviceInfo.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);

	VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = nvvk::make<VkPhysicalDeviceAccelerationStructureFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);

	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

	deviceInfo.addInstanceExtension(VK_KHR_SURFACE_EXTENSION_NAME, false);
	deviceInfo.addInstanceExtension(VK_KHR_WIN32_SURFACE_EXTENSION_NAME, false);

	context.init(deviceInfo);            // Initialize the context
	// Device must support acceleration structures and ray queries:
	assert(asFeatures.accelerationStructure == VK_TRUE && rayQueryFeatures.rayQuery == VK_TRUE);


	/* MEMORY SETUP */

	// Create the allocator
	nvvk::ResourceAllocatorDedicated allocator;
	allocator.init(context, context.m_physicalDevice);

	// Create a buffer
	// TODO: currently a 3-channel 32-bit floating point image
	// Later it would be better to use 4-channel (for transparency) 16-bit (faster, space-efficient) images

	VkDeviceSize bufferSizeBytes = render_width * render_height * 3 * sizeof(float);
	VkBufferCreateInfo bufferCreateInfo = nvvk::make<VkBufferCreateInfo>();
	bufferCreateInfo.size = bufferSizeBytes;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	// VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT means that the CPU can read this buffer's memory.
	// VK_MEMORY_PROPERTY_HOST_CACHED_BIT means that the CPU caches this memory.
	// VK_MEMORY_PROPERTY_HOST_COHERENT_BIT means that the CPU side of cache management
	// is handled automatically, with potentially slower reads/writes.
	nvvk::Buffer buffer = allocator.createBuffer(bufferCreateInfo,                         //
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT       //
		| VK_MEMORY_PROPERTY_HOST_CACHED_BIT  //
		| VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);


	// Create the command pool
	VkCommandPoolCreateInfo cmdPoolInfo = nvvk::make<VkCommandPoolCreateInfo>();
	cmdPoolInfo.queueFamilyIndex = context.m_queueGCT;
	VkCommandPool cmdPool;
	NVVK_CHECK(vkCreateCommandPool(context, &cmdPoolInfo, nullptr, &cmdPool));

	// Load the mesh of the first shape from an OBJ file
	const std::string        exePath(argv[0], std::string(argv[0]).find_last_of("/\\") + 1);
	std::vector<std::string> searchPaths = { exePath + PROJECT_RELDIRECTORY, exePath + PROJECT_RELDIRECTORY "..",
											exePath + PROJECT_RELDIRECTORY "../..", exePath + PROJECT_NAME };
	tinyobj::ObjReader       reader;  // Used to read an OBJ file
	tinyobj::ObjReaderConfig readerConfig;
	readerConfig.mtl_search_path = searchPaths[0];
	reader.ParseFromFile(nvh::findFile("resources/scenes/test_scene_back.obj", searchPaths));
	assert(reader.Valid());  // Make sure tinyobj was able to parse this file

	Model cornellBoxModel{ reader, allocator, context, cmdPool };

	// Describe the bottom-level acceleration structure (BLAS)
	std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;

	blases.push_back(cornellBoxModel.GetBLASInput(context, true));
	// Create the BLAS
	nvvk::RaytracingBuilderKHR raytracingBuilder;
	raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
	// TODO: If BLAS changes needs updates each frame (not static) this should be implemented differently
	// because nvvk::RaytracingBuilderKHR::buildBlas makes CPU wait for the GPU to build the BLAS.
	// Also, if the geometry is static or updated rarely, it's worth to use compaction to save memory.
	raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


	// Create an instance pointing to this BLAS, and build it into a TLAS:
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	{
		// Creating a transform matrix
		nvmath::mat4f transform(1);

		VkAccelerationStructureInstanceKHR instance{};
		instance.accelerationStructureReference = raytracingBuilder.getBlasDeviceAddress(0);  // The address of the BLAS in `blases` that this instance points to
		// Set the instance transform to the identity matrix:
		instance.transform = nvvk::toTransformMatrixKHR(transform);
		instance.instanceCustomIndex = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
		// Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
		instance.instanceShaderBindingTableRecordOffset = 0;
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
		instance.mask = 0xFF;
		instances.push_back(instance);
	}
	
	raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	// Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
	// 0 - a storage buffer (the buffer `buffer`)
	// 1 - an acceleration structure (the TLAS)
	// 2 - storage buffer (for vertex buffer)
	// 3 - storage buffer (for index buffer)
	// 4 - storage buffer (for material ID buffer)
	nvvk::DescriptorSetContainer descriptorSetContainer(context);
	descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

	// Create a layout from the list of bindings
	descriptorSetContainer.initLayout();
	// Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
	descriptorSetContainer.initPool(1);
	// Create a simple pipeline layout from the descriptor set layout:
	descriptorSetContainer.initPipeLayout();

	// Write values into the descriptor set.
	std::array<VkWriteDescriptorSet, 5> writeDescriptorSets;
	// 0
	VkDescriptorBufferInfo descriptorBufferInfo{};
	descriptorBufferInfo.buffer = buffer.buffer;    // The VkBuffer object
	descriptorBufferInfo.range = bufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);
	// 1
	VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
	VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
	descriptorAS.accelerationStructureCount = 1;
	descriptorAS.pAccelerationStructures = &tlasCopy;
	writeDescriptorSets[1] = descriptorSetContainer.makeWrite(0, 1, &descriptorAS);
	// 2
	VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
	vertexDescriptorBufferInfo.buffer = cornellBoxModel.vertexBuffer.buffer;
	vertexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[2] = descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);
	// 3
	VkDescriptorBufferInfo indexDescriptorBufferInfo{};
	indexDescriptorBufferInfo.buffer = cornellBoxModel.indexBuffer.buffer;
	indexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[3] = descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);
	// 4
	VkDescriptorBufferInfo materialDescriptorBufferInfo{};
	materialDescriptorBufferInfo.buffer = cornellBoxModel.materialIdBuffer.buffer;
	materialDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[4] = descriptorSetContainer.makeWrite(0, 4, &materialDescriptorBufferInfo);

	vkUpdateDescriptorSets(context,                                            // The context
		static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
		writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
		0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

	// Shader loading and pipeline creation
	VkShaderModule rayTraceModule =
		nvvk::createShaderModule(context, nvh::loadFile("shaders/volpath.comp.glsl.spv", true, searchPaths));

	// Describes the entrypoint and the stage to use for this shader module in the pipeline
	VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
	shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderStageCreateInfo.module = rayTraceModule;
	shaderStageCreateInfo.pName = "main";

	// Create the compute pipeline
	VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
	pipelineCreateInfo.stage = shaderStageCreateInfo;
	pipelineCreateInfo.layout = descriptorSetContainer.getPipeLayout();

	VkPipeline computePipeline;
	NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
		VK_NULL_HANDLE,          // Pipeline cache (uses default)
		1, &pipelineCreateInfo,  // Compute pipeline create info
		VK_NULL_HANDLE,          // Allocator (uses default)
		&computePipeline));      // Output

	// Create and start recording a command buffer
	VkCommandBuffer cmdBuffer = Utils::AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);


	// Bind the compute shader pipeline
	vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

	// Bind the descriptor set
	VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
	vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, descriptorSetContainer.getPipeLayout(), 0, 1,
		&descriptorSet, 0, nullptr);

	// Run the compute shader with enough workgroups to cover the entire buffer:
	vkCmdDispatch(cmdBuffer, (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
		(uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);

	// Add a command that says "Make it so that memory writes by the vkCmdFillBuffer call
	// are available to read from the CPU." (In other words, "Flush the GPU caches
	// so the CPU can read the data.") To do this, we use a memory barrier.
	VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
	memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;	 // Make shader writes
	memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;       // Readable by the CPU
	vkCmdPipelineBarrier(cmdBuffer,                              // The command buffer
		VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,     // From the transfer stage
		VK_PIPELINE_STAGE_HOST_BIT,               // To the CPU
		0,                                        // No special flags
		1, &memoryBarrier,                        // An array of memory barriers
		0, nullptr, 0, nullptr);                  // No other barriers

	// End and submit the command buffer, then wait for it to finish:
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
	// TODO: vkQueueSubmit is handled by the OS (much slower than Vulkan)
	// Ideally we wanna batch submit command buffers

	// Wait for the GPU to finish
	NVVK_CHECK(vkQueueWaitIdle(context.m_queueGCT));


	// Get the image data back from the GPU
	void* data = allocator.map(buffer);
	stbi_write_hdr("out.hdr", render_width, render_height, 3, reinterpret_cast<float*>(data));
	allocator.unmap(buffer);


	vkDestroyPipeline(context, computePipeline, nullptr);
	vkDestroyShaderModule(context, rayTraceModule, nullptr);
	descriptorSetContainer.deinit();
	raytracingBuilder.destroy();
	allocator.destroy(cornellBoxModel.vertexBuffer);
	allocator.destroy(cornellBoxModel.indexBuffer);
	allocator.destroy(cornellBoxModel.materialIdBuffer);
	vkDestroyCommandPool(context, cmdPool, nullptr);
	allocator.destroy(buffer);
	allocator.deinit();
	context.deinit();                    // Don't forget to clean up at the end of the program!
}