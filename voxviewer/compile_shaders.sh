slang_command="slangc -DFALCOR_MATERIAL_INSTANCE_SIZE=128 -DFALCOR_NVAPI_AVAILABLE=0 -I Falcor/Source/Falcor -O3"
$slang_command src/shaders/blit_srgb.slang -entry VSMain -o assets/shaders/blit_srgb_vs.wgsl
$slang_command src/shaders/blit_srgb.slang -entry PSMain -o assets/shaders/blit_srgb_ps.wgsl
$slang_command src/shaders/raytrace.slang -entry main -o assets/shaders/raytrace.wgsl
