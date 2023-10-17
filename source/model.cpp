#include "model.hpp"

Model::Model(tinyobj::ObjReader reader)
{
	objVertices = reader.GetAttrib().GetVertices();
	const std::vector<tinyobj::shape_t>& objShapes = reader.GetShapes();  // All shapes in the file
	assert(objShapes.size() == 1);                                          // Check that this file has only one shape
	const tinyobj::shape_t& objShape = objShapes[0];                        // Get the first shape

	// Get the indices of the vertices of the first mesh of `objShape` in `attrib.vertices`:
	objIndices.reserve(objShape.mesh.indices.size());
	for (const tinyobj::index_t& index : objShape.mesh.indices)
	{
		objIndices.push_back(index.vertex_index);
	}
}

nvvk::RaytracingBuilderKHR::BlasInput Model::GetBLASInput(
	nvvk::Context& context, nvvk::Buffer& vertexBuffer, nvvk::Buffer& indexBuffer)
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
	geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
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