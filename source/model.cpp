#include "model.hpp"

Model::Model(tinyobj::ObjReader reader,
	nvvk::ResourceAllocatorDedicated& allocator,
	nvvk::Context& context,
	VkCommandPool& cmdPool)
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
	// Start a command buffer for uploading the buffers
	VkCommandBuffer uploadCmdBuffer = Utils::AllocateAndBeginOneTimeCommandBuffer(context, cmdPool);
	// We get these buffers' device addresses, and use them as storage buffers and build inputs.
	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
		| VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
	vertexBuffer = allocator.createBuffer(uploadCmdBuffer, objVertices, usage);
	indexBuffer = allocator.createBuffer(uploadCmdBuffer, objIndices, usage);
	materialIdBuffer = allocator.createBuffer(uploadCmdBuffer, objMaterials, usage);
	Utils::EndSubmitWaitAndFreeCommandBuffer(context, context.m_queueGCT, cmdPool, uploadCmdBuffer);
	allocator.finalizeAndReleaseStaging();
}

nvvk::RaytracingBuilderKHR::BlasInput Model::GetBLASInput(nvvk::Context& context, bool isOpaque)
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
	if (isOpaque) geometry.flags = 0;
	else geometry.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
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