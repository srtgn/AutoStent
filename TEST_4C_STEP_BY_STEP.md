# Step-by-Step Guide: Testing 4C Docker Image

## Answer to Your Questions

### Q1: Do we have YAML files for each stent geometry?
**Answer: NO** - Currently, the code generates YAML files on-the-fly, but they're incomplete because:
- `MESH_TOOLS_AVAILABLE = False` (mesh_generator.py import fails)
- Without mesh tools, we can't generate proper geometry
- The generated YAML is missing the `STRUCTURE GEOMETRY` section that 4C requires

### Q2: Can we use 4C Docker image instructions?
**Answer: YES!** The 4C Docker image comes with test files we can use.

---

## Step-by-Step Testing Instructions

### Method 1: Test Locally (Recommended First)

#### Step 1: Run the Test Script
```bash
cd /Users/srtgn/Documents/Projects/4c
./railway/test_4c_docker.sh
```

This script will:
- Check Docker is installed
- Pull the 4C Docker image
- Test the 4C binary
- List available test files
- Run a test simulation

#### Step 2: Check Results
The script will show:
- ✅ If 4C runs successfully
- ❌ Error messages if it fails
- Output files created

### Method 2: Test via API Endpoint

#### Step 1: Start the Railway server
```bash
# The server should already be running at https://4c.up.railway.app
```

#### Step 2: Call the test endpoint
```bash
curl https://4c.up.railway.app/test-4c-docker
```

This will:
- Run 4C Docker image with a tutorial file
- Return detailed results including stdout/stderr
- Show what output files were created

### Method 3: Manual Docker Test

#### Step 1: Pull the image
```bash
docker pull ghcr.io/4c-multiphysics/4c:main
```

#### Step 2: List test files
```bash
docker run --rm ghcr.io/4c-multiphysics/4c:main \
  find /home/user/4C/tests/input_files -name "*.4C.yaml" -type f
```

#### Step 3: Run a test simulation
```bash
# Create output directory
mkdir -p /tmp/4c_test_output

# Run 4C with tutorial file
docker run --rm \
  -v /tmp/4c_test_output:/output \
  -w /output \
  ghcr.io/4c-multiphysics/4c:main \
  /home/user/4C/build/4C \
  /home/user/4C/tests/input_files/tutorial_solid_vtu.4C.yaml \
  test_output

# Check results
ls -lh /tmp/4c_test_output
```

---

## What 4C Actually Needs

Based on the tutorial file (`tutorial_solid_vtu.4C.yaml`), 4C requires:

1. **YAML Format** (correct format):
   ```yaml
   TITLE: Stent simulation
   PROBLEM TYPE:
     PROBLEMTYPE: Structure
   
   STRUCTURE GEOMETRY:
     FILE: geometry.vtu  # ← Needs a VTU mesh file!
     ELEMENT_BLOCKS:
       - ID: 1
         SOLID:
           HEX8:
             MAT: 1
   ```

2. **VTU Geometry File**: 4C expects a VTU (VTK unstructured) mesh file, not inline geometry

3. **Proper Material Definition**:
   ```yaml
   MATERIALS:
     - MAT: 1
       MAT_Struct_StVenantKirchhoff:
         YOUNG: 200000.0
         NUE: 0.3
         DENS: 6.45e-9
   ```

---

## Why Our Current Code Fails

1. **No VTU file generation**: We try to write geometry inline, but 4C expects a VTU file reference
2. **Wrong YAML format**: We use `PROBLEM_TYPE:` but 4C expects `PROBLEM TYPE:` (with space)
3. **Missing mesh tools**: `mesh_generator.py` import fails, so we can't generate meshes

---

## Solutions to Make 4C Work

### Solution 1: Use VTU File Generation (Best)
- Generate VTU mesh files using `pyvista` or similar
- Save VTU file alongside YAML
- Reference it in YAML: `FILE: geometry.vtu`

### Solution 2: Use 4C's Built-in Geometry (Simpler)
- Use 4C's geometry generation capabilities
- Or use a pre-generated simple mesh

### Solution 3: Fix Mesh Tools Import
- Ensure `railway/mesh_generator.py` is in Docker image
- Fix import path issues
- Generate proper VTU files

---

## Next Steps

1. **Run the test script** to verify 4C Docker image works
2. **Check the `/test-4c-docker` endpoint** to see what 4C actually outputs
3. **Fix YAML generation** to match 4C's format
4. **Generate VTU files** instead of inline geometry

Would you like me to:
- Fix the YAML generator to use VTU files?
- Create a VTU mesh generator?
- Test the 4C Docker image now?



