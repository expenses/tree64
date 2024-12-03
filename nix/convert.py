import bpy
import sys


def main():
    # Parse command-line arguments
    argv = sys.argv
    if "--" not in argv:
        print("Error: Script requires arguments after '--'")
        return
    argv = argv[argv.index("--") + 1 :]  # Get all args after '--'

    if len(argv) != 2:
        print(
            "Usage: blender --background --python usd_to_gltf.py -- <input_usd> <output_gltf>"
        )
        return

    input_usd = argv[0]
    output_gltf = argv[1]

    # Import USD file
    try:
        bpy.ops.wm.read_factory_settings(use_empty=True)  # Clear the scene
        bpy.ops.wm.usd_import(filepath=input_usd)
        print(f"Successfully imported USD file: {input_usd}")
    except Exception as e:
        print(f"Error importing USD file: {e}")
        return

    # Export to GLTF
    try:
        bpy.ops.export_scene.gltf(filepath=output_gltf, export_format="GLB")
        print(f"Successfully exported to GLTF file: {output_gltf}")
    except Exception as e:
        print(f"Error exporting GLTF file: {e}")
        return


if __name__ == "__main__":
    main()
