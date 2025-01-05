slangc -DFALCOR_MATERIAL_INSTANCE_SIZE=128 -I Falcor/Source/Falcor -O3 src/shaders/blit_srgb.slang -entry VSMain -o assets/shaders/blit_srgb_vs.wgsl
slangc -DFALCOR_MATERIAL_INSTANCE_SIZE=128 -I Falcor/Source/Falcor -O3 src/shaders/blit_srgb.slang -entry PSMain -o assets/shaders/blit_srgb_ps.wgsl
slangc -DFALCOR_MATERIAL_INSTANCE_SIZE=128 -I Falcor/Source/Falcor -O3 src/shaders/raytrace.slang -entry main -o assets/shaders/raytrace.wgsl
