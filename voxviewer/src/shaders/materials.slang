struct Material {
    float3 base_colour;
    float emission_factor;
    float linear_roughness;
    float metallic;

    property float specular_factor {
        get {
            return 1.0;
        }
    };

    property float3 specular_colour {
        get {
            return float3(1.0);
        }
    };

    // Defaults to 1.5
    // https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#dielectrics
    property float ior {
        get {
            return 1.5;
        }
    };

    float f0_from_ior() {
        let root = (1-ior)/(1+ior);
        return root * root;
    }

    // Reflections don't show if a material has exactly 0 roughness.
    float alpha_roughness() {
        return max(linear_roughness * linear_roughness, 0.000001);
    }

    float3 dielectic_f0() {
        return f0_from_ior() * specular_colour * specular_factor;
    }

    float3 f0() {
        return lerp(dielectic_f0(), base_colour, metallic);
    }

    float f90() {
        return lerp(specular_factor, 1.0, metallic);
    }

    float3 diffuse_colour() {
        return lerp(base_colour, float3(0.0), metallic);
    }
};
