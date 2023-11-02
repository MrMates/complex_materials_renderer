#include <cassert>
#include <array>
#include <iostream>
#include <random>

#include <nvvk/context_vk.hpp>
#include <nvvk/sbtwrapper_vk.hpp>
#include <nvvk/structs_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>
#include <nvvk/error_vk.hpp>
#include <nvvk/shaders_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>
#include <nvh/fileoperations.hpp>

#include <model.hpp>
#include <utils.hpp>
#include <shaders/common.hpp>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>


static const uint64_t render_width = 1920;
static const uint64_t render_height = 1080;

static const uint64_t workgroup_width = 16;
static const uint64_t workgroup_height = 8;

struct SBTDiffuseData {
	VkDeviceAddress vertexBuffer;
	VkDeviceAddress indexBuffer;
};

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo addressInfo = nvvk::make<VkBufferDeviceAddressInfo>();
	addressInfo.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &addressInfo);
}

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

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures = nvvk::make<VkPhysicalDeviceRayTracingPipelineFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeatures);

	// TODO: delete after ray tracing is implemented
	VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures = nvvk::make<VkPhysicalDeviceRayQueryFeaturesKHR>();
	deviceInfo.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);
	// --

	context.init(deviceInfo);            // Initialize the context
	// Device must support acceleration structures and ray queries:
	assert(asFeatures.accelerationStructure == VK_TRUE && rtPipelineFeatures.rayTracingPipeline == VK_TRUE);

	// Requesting physical device raytracing properties
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties = nvvk::make<VkPhysicalDeviceRayTracingPipelinePropertiesKHR>();
	VkPhysicalDeviceProperties2 prop2 = nvvk::make<VkPhysicalDeviceProperties2>();
	prop2.pNext = &rtProperties;
	vkGetPhysicalDeviceProperties2(context.m_physicalDevice, &prop2);


	/* MEMORY SETUP */

	// Create the allocator
	nvvk::ResourceAllocatorDedicated allocator;
	allocator.init(context, context.m_physicalDevice);

	// Setup the SBT
	nvvk::SBTWrapper sbtWrapper{};
	sbtWrapper.setup(context, 0, &allocator, rtProperties);


	// Create a buffer
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
	reader.ParseFromFile(nvh::findFile("resources/scenes/cube.obj", searchPaths));
	assert(reader.Valid());  // Make sure tinyobj was able to parse this file

	Model model { reader, allocator, context, cmdPool };

	reader.ParseFromFile(nvh::findFile("resources/scenes/box.obj", searchPaths));
	assert(reader.Valid());  // Make sure tinyobj was able to parse this file

	Model boxModel{ reader, allocator, context, cmdPool };

	// Describe the bottom-level acceleration structure (BLAS)
	std::vector<nvvk::RaytracingBuilderKHR::BlasInput> blases;

	blases.push_back(model.GetBLASInput(context, true));
	blases.push_back(boxModel.GetBLASInput(context, true));

	// Create the BLAS
	nvvk::RaytracingBuilderKHR raytracingBuilder;
	raytracingBuilder.setup(context, &allocator, context.m_queueGCT);
	// TODO: If BLAS changes needs updates each frame (not static) this should be implemented differently
	// because nvvk::RaytracingBuilderKHR::buildBlas makes CPU wait for the GPU to build the BLAS.
	// Also, if the geometry is static or updated rarely, it's worth to use compaction to save memory.
	raytracingBuilder.buildBlas(blases, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


	// Create an instance pointing to this BLAS, and build it into a TLAS:
	std::vector<VkAccelerationStructureInstanceKHR> instances;
	std::default_random_engine randEngine;
	std::uniform_real_distribution<float> uniformDist(-.5f, .5f);
	for (int x = 0; x <= 1; x++)
	{
		for (int y = 0; y <= 1; y++)
		{
			// Creating a transform matrix
			nvmath::mat4f transform(1);

			transform.translate(nvmath::vec3f(float(x), float(y) + 2.f, 0.0f));
			transform.scale(1.0 / 3.0);
			//transform.rotate(uniformDist(randEngine), nvmath::vec3f(0.0f, 1.0f, 0.0f));
			//transform.rotate(uniformDist(randEngine), nvmath::vec3f(1.0f, 0.0f, 0.0f));

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
		
	}

	{
		// Blas for the box
		// Creating a transform matrix
		nvmath::mat4f transform(1);
		transform.translate(nvmath::vec3f(0.0f, -1.f, -50.0f));

		VkAccelerationStructureInstanceKHR instance{};
		instance.accelerationStructureReference = raytracingBuilder.getBlasDeviceAddress(1);  // The address of the BLAS in `blases` that this instance points to
		// Set the instance transform to the identity matrix:
		instance.transform = nvvk::toTransformMatrixKHR(transform);
		instance.instanceCustomIndex = 0;  // 24 bits accessible to ray shaders via rayQueryGetIntersectionInstanceCustomIndexEXT
		// Used for a shader offset index, accessible via rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT
		instance.instanceShaderBindingTableRecordOffset = 0; // Offset to make it use the diffuse shader
		instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;  // How to trace this instance
		instance.mask = 0xFF;
		instances.push_back(instance);
	}

	raytracingBuilder.buildTlas(instances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


	nvvk::DescriptorSetContainer descriptorSetContainer(context);
	descriptorSetContainer.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);
	descriptorSetContainer.addBinding(RtxBindings::eOutBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_RAYGEN_BIT_KHR);


	// Create a layout from the list of bindings
	descriptorSetContainer.initLayout();
	// Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
	descriptorSetContainer.initPool(1);
	// TODO: Only basic pipeline layout
	descriptorSetContainer.initPipeLayout();


	// Write values into the descriptor set.
	std::array<VkWriteDescriptorSet, 2> writeDescriptorSets;

	VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
	VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
	descriptorAS.accelerationStructureCount = 1;
	descriptorAS.pAccelerationStructures = &tlasCopy;
	writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0, RtxBindings::eTlas, &descriptorAS);

	VkDescriptorBufferInfo descriptorBufferInfo{};
	descriptorBufferInfo.buffer = buffer.buffer;    // The VkBuffer object
	descriptorBufferInfo.range = bufferSizeBytes;  // The length of memory to bind; offset is 0.
	writeDescriptorSets[1] = descriptorSetContainer.makeWrite(0, RtxBindings::eOutBuffer, &descriptorBufferInfo);

	vkUpdateDescriptorSets(context,                                            // The context
		static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
		writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
		0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)


	enum StageIndices
	{
		eRaygen,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};

	// All stages
	std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
	VkPipelineShaderStageCreateInfo              stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
	stage.pName = "main";  // All the same entry point
	// Raygen
	stage.module = nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.rgen.glsl.spv", true, searchPaths));
	stage.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[eRaygen] = stage;
	// Miss
	stage.module = nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.rmiss.glsl.spv", true, searchPaths));
	stage.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[eMiss] = stage;
	// Hit Group 1 - Closest Hit
	stage.module = nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.rchit.glsl.spv", true, searchPaths));
	stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	stages[eClosestHit] = stage;
	//// Hit Group 1 - Any Hit
	//stage.module = nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.rahit.glsl.spv", true, searchPaths));
	//stage.stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	//stages[eAnyHit] = stage;
	//// Hit Group 2 - Closest Hit
	//stage.module = nvvk::createShaderModule(context, nvh::loadFile("shaders/diffuse.rchit.glsl.spv", true, searchPaths));
	//stage.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	//stages[eDiffuseCHit] = stage;

	// Shader groups
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	// Raygen
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	shaderGroups.push_back(group);

	// Miss
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	shaderGroups.push_back(group);

	// Hit group for media
	// TODO: For volumetric media later, change this from triangles to procedural
	// And write an intersection shader
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	//group.anyHitShader = eAnyHit;
	shaderGroups.push_back(group);

	// Hit group for the scene (background box)
	//group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	//group.generalShader = VK_SHADER_UNUSED_KHR;
	//group.closestHitShader = eDiffuseCHit;
	//shaderGroups.push_back(group);

	// Assemble the shader stages and recursion depth info into the ray tracing pipeline
	VkRayTracingPipelineCreateInfoKHR rayPipelineInfo = nvvk::make<VkRayTracingPipelineCreateInfoKHR>();
	rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
	rayPipelineInfo.pStages = stages.data();
	// In this case, m_rtShaderGroups.size() == 3: we have one raygen group,
	// one miss shader group, and one hit group.
	rayPipelineInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
	rayPipelineInfo.pGroups = shaderGroups.data();
	rayPipelineInfo.maxPipelineRayRecursionDepth = 1;  // Ray depth
	rayPipelineInfo.layout = descriptorSetContainer.getPipeLayout();

	VkPipeline rayTracingPipeline;
	NVVK_CHECK(vkCreateRayTracingPipelinesKHR(context, {}, {}, 1, &rayPipelineInfo, nullptr, &rayTracingPipeline));
	for (auto& s : stages)
	{
		vkDestroyShaderModule(context, s.module, nullptr);
	}

	/* COMPUTE SHADER BINDINGS */

	//// Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl:
	//// 0 - a storage buffer (the buffer `buffer`)
	//// 1 - an acceleration structure (the TLAS)
	//// 2 - storage buffer (for vertex buffer)
	//// 3 - storage buffer (for index buffer)
	//nvvk::DescriptorSetContainer descriptorSetContainer(context);
	//descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	//descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	//descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);
	//descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT);

	//// Create a layout from the list of bindings
	//descriptorSetContainer.initLayout();
	//// Create a descriptor pool from the list of bindings with space for 1 set, and allocate that set
	//descriptorSetContainer.initPool(1);
	//// Create a simple pipeline layout from the descriptor set layout:
	//descriptorSetContainer.initPipeLayout();

	//// Write values into the descriptor set.
	//std::array<VkWriteDescriptorSet, 4> writeDescriptorSets;
	//// 0
	//VkDescriptorBufferInfo descriptorBufferInfo{};
	//descriptorBufferInfo.buffer = buffer.buffer;    // The VkBuffer object
	//descriptorBufferInfo.range = bufferSizeBytes;  // The length of memory to bind; offset is 0.
	//writeDescriptorSets[0] = descriptorSetContainer.makeWrite(0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);
	//// 1
	//VkWriteDescriptorSetAccelerationStructureKHR descriptorAS = nvvk::make<VkWriteDescriptorSetAccelerationStructureKHR>();
	//VkAccelerationStructureKHR tlasCopy = raytracingBuilder.getAccelerationStructure();  // So that we can take its address
	//descriptorAS.accelerationStructureCount = 1;
	//descriptorAS.pAccelerationStructures = &tlasCopy;
	//writeDescriptorSets[1] = descriptorSetContainer.makeWrite(0, 1, &descriptorAS);
	//// 2
	//VkDescriptorBufferInfo vertexDescriptorBufferInfo{};
	//vertexDescriptorBufferInfo.buffer = vertexBuffer.buffer;
	//vertexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	//writeDescriptorSets[2] = descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);
	//// 3
	//VkDescriptorBufferInfo indexDescriptorBufferInfo{};
	//indexDescriptorBufferInfo.buffer = indexBuffer.buffer;
	//indexDescriptorBufferInfo.range = VK_WHOLE_SIZE;
	//writeDescriptorSets[3] = descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);

	//vkUpdateDescriptorSets(context,                                            // The context
	//	static_cast<uint32_t>(writeDescriptorSets.size()),  // Number of VkWriteDescriptorSet objects
	//	writeDescriptorSets.data(),                         // Pointer to VkWriteDescriptorSet objects
	//	0, nullptr);  // An array of VkCopyDescriptorSet objects (unused)

	//// Shader loading and pipeline creation
	//VkShaderModule rayTraceModule =
	//	nvvk::createShaderModule(context, nvh::loadFile("shaders/raytrace.comp.glsl.spv", true, searchPaths));

	//// Describes the entrypoint and the stage to use for this shader module in the pipeline
	//VkPipelineShaderStageCreateInfo shaderStageCreateInfo = nvvk::make<VkPipelineShaderStageCreateInfo>();
	//shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	//shaderStageCreateInfo.module = rayTraceModule;
	//shaderStageCreateInfo.pName = "main";

	//// Create the compute pipeline
	//VkComputePipelineCreateInfo pipelineCreateInfo = nvvk::make<VkComputePipelineCreateInfo>();
	//pipelineCreateInfo.stage = shaderStageCreateInfo;
	//pipelineCreateInfo.layout = descriptorSetContainer.getPipeLayout();

	//VkPipeline computePipeline;
	//NVVK_CHECK(vkCreateComputePipelines(context,                 // Device
	//	VK_NULL_HANDLE,          // Pipeline cache (uses default)
	//	1, &pipelineCreateInfo,  // Compute pipeline create info
	//	VK_NULL_HANDLE,          // Allocator (uses default)
	//	&computePipeline));      // Output


	/* END COMPUTE SHADER BINDINGS */

	// Create and start recording a command buffer
	VkCommandBuffer cmdBuffer = Utils::AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
	
	// Bind the compute shader pipeline
	vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipeline);

	// Bind the descriptor set
	VkDescriptorSet descriptorSet = descriptorSetContainer.getSet(0);
	vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, descriptorSetContainer.getPipeLayout(), 0, 1,
		&descriptorSet, 0, nullptr);

	sbtWrapper.addData(nvvk::SBTWrapper::eHit,
		0,
		SBTDiffuseData{
			GetBufferDeviceAddress(context, boxModel.vertexBuffer.buffer),
			GetBufferDeviceAddress(context, boxModel.indexBuffer.buffer),
		});

	sbtWrapper.create(rayTracingPipeline, rayPipelineInfo);
	auto& regions = sbtWrapper.getRegions();
	vkCmdTraceRaysKHR(cmdBuffer, &regions[0], &regions[1], &regions[2], &regions[3], render_width, render_height, 1);

	//// Run the compute shader with enough workgroups to cover the entire buffer:
	//vkCmdDispatch(cmdBuffer, (uint32_t(render_width) + workgroup_width - 1) / workgroup_width,
	//	(uint32_t(render_height) + workgroup_height - 1) / workgroup_height, 1);



	//// Add a command that says "Make it so that memory writes by the vkCmdFillBuffer call
	//// are available to read from the CPU." (In other words, "Flush the GPU caches
	//// so the CPU can read the data.") To do this, we use a memory barrier.
	//VkMemoryBarrier memoryBarrier = nvvk::make<VkMemoryBarrier>();
	//memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;	 // Make shader writes
	//memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;       // Readable by the CPU
	//vkCmdPipelineBarrier(cmdBuffer,                              // The command buffer
	//	VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,     // From the transfer stage
	//	VK_PIPELINE_STAGE_HOST_BIT,               // To the CPU
	//	0,                                        // No special flags
	//	1, &memoryBarrier,                        // An array of memory barriers
	//	0, nullptr, 0, nullptr);                  // No other barriers

	// End and submit the command buffer, then wait for it to finish:
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, cmdBuffer);
	// TODO: vkQueueSubmit is handled by the OS (much slower than Vulkan)
	// Ideally we wanna batch submit command buffers
	
	// Get the image data back from the GPU
	void* data = allocator.map(buffer);
	stbi_write_hdr("out.hdr", render_width, render_height, 3, reinterpret_cast<float*>(data));
	allocator.unmap(buffer);


	sbtWrapper.destroy();
	vkDestroyPipeline(context, rayTracingPipeline, nullptr);
	descriptorSetContainer.deinit();
	raytracingBuilder.destroy();
	allocator.destroy(model.vertexBuffer);
	allocator.destroy(model.indexBuffer);
	allocator.destroy(boxModel.vertexBuffer);
	allocator.destroy(boxModel.indexBuffer);
	vkDestroyCommandPool(context, cmdPool, nullptr);
	allocator.destroy(buffer);
	allocator.deinit();
	context.deinit();                    // Don't forget to clean up at the end of the program!
}