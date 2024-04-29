#include "model.hpp"
#include "json.hpp"
using json = nlohmann::ordered_json;

#include <fstream>
#include <iostream>

Model::Model(tinyobj::ObjReader reader,
	nvvk::ResourceAllocatorDedicated& allocator,
	nvvk::Context& context,
	VkCommandPool& cmdPool,
	std::string filepath,
	Options* options)
{
	objVertices = reader.GetAttrib().GetVertices();
	const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file

	for (size_t s = 0; s < objShapes.size(); s++)
	{
		const tinyobj::shape_t& objShape = objShapes[s];                        // Get the first shape

		// Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
		objIndices.reserve(objIndices.size() + objShape.mesh.indices.size());
		for (const tinyobj::index_t& index : objShape.mesh.indices)
		{
			objIndices.push_back(index.vertex_index);
		}

		objMaterials.reserve(objMaterials.size() + objShape.mesh.material_ids.size());
		for (const int& id : objShape.mesh.material_ids)
		{
			objMaterials.push_back(id);
		}
	}

	std::string jsonFileName = filepath.substr(0, filepath.find_last_of('.')).append(".json");
	std::ifstream f(jsonFileName);
	assert(f.is_open() && "Media definition .json file not found. Run mat_parser.py for this .obj scene first.");
	auto data = nlohmann::ordered_json::parse(f);

	// Buffer in format count, (id, vec3(sigma_s), vec3(sigma_a), vec3(g), ior)*count
	mediaDefinitions.push_back(data.size());
	for (auto it = data.begin(); it != data.end(); ++it)
	{
		// Scene definitions
		if (it.key() == "scene")
		{
			auto& cameraPos = it.value()["camera"];
			for (size_t i = 0; i < 3; i++)
			{
				options->cameraPos[i] = std::stof(cameraPos[i].dump());
			}
			auto& cameraLookAt = it.value()["cameraLookAt"];
			for (size_t i = 0; i < 3; i++)
			{
				options->cameraLookAt[i] = std::stof(cameraLookAt[i].dump());
			}
			auto& lightPos = it.value()["lightPos"];
			for (size_t i = 0; i < 3; i++)
			{
				options->lightPos[i] = std::stof(lightPos[i].dump());
			}
			auto& lightColor = it.value()["lightColor"];
			for (size_t i = 0; i < 3; i++)
			{
				options->lightColor[i] = std::stof(lightColor[i].dump());
			}
			options->cameraFOV = it.value()["fov"];
			options->lightIntensity = it.value()["lightIntensity"];
			options->scale = it.value()["scale"];
			continue;
		}

		mediaDefinitions.push_back(std::stof(it.key()));
		auto& sigmaS = it.value()["sigma_s"];
		for (size_t i = 0; i < 3; i++)
		{
			mediaDefinitions.push_back(std::stof(sigmaS[i].dump()));
		}

		auto& sigmaA = it.value()["sigma_a"];
		for (size_t i = 0; i < 3; i++)
		{
			mediaDefinitions.push_back(std::stof(sigmaA[i].dump()));
		}

		auto& g = it.value()["g"];
		for (size_t i = 0; i < 3; i++)
		{
			mediaDefinitions.push_back(std::stof(g[i].dump()));
		}

		auto& ior = it.value()["ior"];
		mediaDefinitions.push_back(std::stof(ior.dump()));
	}

	f.close();

	// Start a command buffer for uploading the buffers
	VkCommandBuffer uploadCmdBuffer = Utils::AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
	// We get these buffers' device addresses, and use them as storage buffers and build inputs.
	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
	indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);
	materialIdBuffer = allocator.createBuffer(uploadCmdBuffer, objMaterials, usage);
	mediaDefinitionsBuffer = allocator.createBuffer(uploadCmdBuffer, mediaDefinitions, usage);
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
	allocator.finalizeAndReleaseStaging();
}

nvvk::RaytracingBuilderKHR::BlasInput Model::GetBLASInput(nvvk::Context& context)
{
	nvvk::RaytracingBuilderKHR::BlasInput blas;

	// Get the device addresses of the vertex and index buffers
	VkDeviceAddress vertexBufferAddress = GetBufferDeviceAddress(context, vertexBuffer.buffer);
	VkDeviceAddress indexBufferAddress = GetBufferDeviceAddress(context, indexBuffer.buffer);

	// Specify where the builder can find the vertices and indices for triangles, and their formats:
	VkAccelerationStructureGeometryTrianglesDataKHR triangles = nvvk::make<VkAccelerationStructureGeometryTrianglesDataKHR>();
	triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	triangles.vertexData.deviceAddress = vertexBufferAddress;
	triangles.vertexStride = 3 * sizeof(float);
	triangles.maxVertex = static_cast<uint32_t>(objVertices.size() / 3 - 1);
	triangles.indexType = VK_INDEX_TYPE_UINT32;
	triangles.indexData.deviceAddress = indexBufferAddress;
	triangles.transformData.deviceAddress = 0;  // No transform

	// Create a VkAccelerationStructureGeometryKHR object that says it handles opaque triangles and points to the above:
	VkAccelerationStructureGeometryKHR geometry = nvvk::make<VkAccelerationStructureGeometryKHR>();
	geometry.geometry.triangles = triangles;
	geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geometry.flags = 0;
	blas.asGeometry.emplace_back(geometry);

	// Create offset info that allows us to say how many triangles and vertices to read
	VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
	offsetInfo.firstVertex = 0;
	offsetInfo.primitiveCount = static_cast<uint32_t>(objIndices.size() / 3);  // Number of triangles
	offsetInfo.primitiveOffset = 0;
	offsetInfo.transformOffset = 0;
	blas.asBuildOffsetInfo.emplace_back(offsetInfo);

	return blas;
}

VkDeviceAddress Model::GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
	VkBufferDeviceAddressInfo addressInfo = nvvk::make<VkBufferDeviceAddressInfo>();
	addressInfo.buffer = buffer;
	return vkGetBufferDeviceAddress(device, &addressInfo);
}