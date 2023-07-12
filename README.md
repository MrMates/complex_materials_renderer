# Complex Materials Renderer
This is source code for my bachelor thesis on "Complex Materials Rendering". It's a renderer written for Vulkan in C++, leveraging basic shading, path tracing and BSSRDF.

## Dependencies
This project uses [nvpro_core](https://github.com/nvpro-samples/nvpro_core) library from NVIDIA. This package needs to be cloned separately.

## Setup
Clone the required repositories and the renderer project itself
```
git clone --recursive --shallow-submodules https://github.com/nvpro-samples/nvpro_core.git
git clone https://github.com/MrMates/complex_materials_renderer.git
```
and then use CMake to build the renderer source files.
