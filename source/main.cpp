#include <cassert>
#include <array>
#include <iostream>
#include <random>
#include <string>

#include <nvvk/context_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/images_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>
#include <nvh/fileoperations.hpp>

#include "model.hpp"
#include "utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
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

	// Image for the GPU
	VkImageCreateInfo imageCreateInfo = nvvk::make<VkImageCreateInfo>();
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.format = VK_FORMAT_R32G32B32A32_SFLOAT; // has to be RGBA because of GPU optimizations
	imageCreateInfo.extent = { render_width, render_height, 1 };
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	imageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	nvvk::Image image = allocator.createImage(imageCreateInfo);

	VkImageViewCreateInfo imageViewCreateInfo = nvvk::make<VkImageViewCreateInfo>();
	imageViewCreateInfo.image = image.image;
	imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	imageViewCreateInfo.format = imageCreateInfo.format;
	imageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	imageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	imageViewCreateInfo.subresourceRange.layerCount = 1;
	imageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	imageViewCreateInfo.subresourceRange.levelCount = 1;
	VkImageView imageView;
	NVVK_CHECK(vkCreateImageView(context, &imageViewCreateInfo, nullptr, &imageView));

	// GPU output image counterpart for the CPU (to copy to)
	imageCreateInfo.tiling = VK_IMAGE_TILING_LINEAR;
	imageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	nvvk::Image imageLinear = allocator.createImage(imageCreateInfo,
		VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);

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
	std::string objFilePath("resources/scenes/test_scene_back.obj");
	if (argc == 2)
	{
		objFilePath = argv[1];
	}
	reader.ParseFromFile(nvh::findFile(objFilePath, searchPaths));
	assert(reader.Valid());  // Make sure tinyobj was able to parse this file

	Model cornellBoxModel{ reader, allocator, context, cmdPool, nvh::findFile(objFilePath, searchPaths) };

	VkCommandBuffer imageCmdBuffer = Utils::AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
	const VkAccessFlags srcAccesses = 0; // Images don't have a layout (not accessible)
	const VkAccessFlags dstImageAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
	const VkAccessFlags dstImageLinearAccesses = VK_ACCESS_TRANSFER_WRITE_BIT;

	const VkPipelineStageFlags srcStages = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
	const VkPipelineStageFlags dstStages = nvvk::makeAccessMaskPipelineStageFlags(dstImageAccesses | dstImageLinearAccesses);

	VkImageMemoryBarrier imageBarriers[2];

	imageBarriers[0] = nvvk::makeImageMemoryBarrier(image.image,
		srcAccesses, dstImageAccesses,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
		VK_IMAGE_ASPECT_COLOR_BIT);

	imageBarriers[1] = nvvk::makeImageMemoryBarrier(imageLinear.image,
		srcAccesses, dstImageLinearAccesses,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		VK_IMAGE_ASPECT_COLOR_BIT);

	vkCmdPipelineBarrier(imageCmdBuffer, srcStages, dstStages, 0, 0, nullptr, 0, nullptr, 2, imageBarriers);
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, imageCmdBuffer);
	allocator.finalizeAndReleaseStaging();

	// Describe the bottom-level acceleration structure (BLAS)
	std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;

	blases.push_back(cornellBoxModel.GetBLASInput(context, true));
	// Create the BLAS
	nvvk::RaytracingBuilderKHR raytracingBuilder;
	raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
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
	// 0 - a storage image (the image `image`)
	// 1 - an acceleration structure (the TLAS)
	// 2 - storage buffer (for vertex buffer)
	// 3 - storage buffer (for index buffer)
	// 4 - storage buffer (for material ID buffer)
	// 5 - storage buffer (for media definitions)
	nvvk::DescriptorSetContainer descriptorSetContainer(context);
	descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	descriptorSetContainer.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

	// Create a layout from the list of bindings
	descriptorSetContainer.initLayout();
	// Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
	descriptorSetContainer.initPool(1);
	// Create a simple pipeline layout from the descriptor set layout:
	descriptorSetContainer.initPipeLayout();

	// Write values into the descriptor set.
	std::array<VkWriteDescriptorSet, 6> writeDescriptorSets;
	// 0
	VkDescriptorImageInfo descriptorImageInfo{};
	descriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
	descriptorImageInfo.imageView = imageView;
	writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0, 0, &descriptorImageInfo);
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
	// 5
	VkDescriptorBufferInfo mediaDescriptorBufferInfo{};
	mediaDescriptorBufferInfo.buffer = cornellBoxModel.mediaDefinitionsBuffer.buffer;
	mediaDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	writeDescriptorSets[5] = descriptorSetContainer.makeWrite(0, 5, &mediaDescriptorBufferInfo);

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

	{
		// We need to change the GPU image layout so it can be read by transfer (copy to the CPU)
		const VkAccessFlags srcAccesses = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
		const VkAccessFlags dstAccesses = VK_ACCESS_TRANSFER_READ_BIT;
		const VkPipelineStageFlags srcStages = nvvk::makeAccessMaskPipelineStageFlags(srcAccesses);
		const VkPipelineStageFlags dstStages = nvvk::makeAccessMaskPipelineStageFlags(dstAccesses);

		const VkImageMemoryBarrier barrier =
			nvvk::makeImageMemoryBarrier(image.image,
				srcAccesses, dstAccesses,
				VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
				VK_IMAGE_ASPECT_COLOR_BIT);
		vkCmdPipelineBarrier(cmdBuffer, srcStages, dstStages, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	}

	// Copy the GPU image to CPU
	VkImageCopy region;
	region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.srcSubresource.baseArrayLayer = 0;
	region.srcSubresource.layerCount = 1;
	region.srcSubresource.mipLevel = 0;

	region.srcOffset = { 0,0,0 };
	region.dstSubresource = region.srcSubresource;
	region.dstOffset = { 0,0,0 };

	region.extent = { render_width, render_height, 1 };
	vkCmdCopyImage(cmdBuffer,
		image.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		imageLinear.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		1, &region);

	// Add a command that says "Make it so that memory writes by the vkCmdFillBuffer call
	// are available to read from the CPU." (In other words, "Flush the GPU caches
	// so the CPU can read the data.") To do this, we use a memory barrier.
	VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
	memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;	 // Make transfer writes
	memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;       // Readable by the CPU
	vkCmdPipelineBarrier(cmdBuffer,                              // The command buffer
		VK_PIPELINE_STAGE_TRANSFER_BIT,			  // From transfers
		VK_PIPELINE_STAGE_HOST_BIT,               // To the CPU
		0,                                        // No special flags
		1, &memoryBarrier,                        // An array of memory barriers
		0, nullptr, 0, nullptr);                  // No other barriers

	// End and submit the command buffer, then wait for it to finish:
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);

	// Wait for the GPU to finish
	NVVK_CHECK(vkQueueWaitIdle(context.m_queueGCT));


	// Get the image data back from the GPU
	void* data = allocator.map(imageLinear);
	stbi_write_hdr("out.hdr", render_width, render_height, 4, reinterpret_cast<float*>(data));
	allocator.unmap(imageLinear);


	vkDestroyPipeline(context, computePipeline, nullptr);
	vkDestroyShaderModule(context, rayTraceModule, nullptr);
	descriptorSetContainer.deinit();
	raytracingBuilder.destroy();
	allocator.destroy(cornellBoxModel.vertexBuffer);
	allocator.destroy(cornellBoxModel.indexBuffer);
	allocator.destroy(cornellBoxModel.materialIdBuffer);
	vkDestroyCommandPool(context, cmdPool, nullptr);
	allocator.destroy(imageLinear);
	vkDestroyImageView(context, imageView, nullptr);
	allocator.destroy(image);
	allocator.deinit();
	context.deinit();                    // Don't forget to clean up at the end of the program!
}