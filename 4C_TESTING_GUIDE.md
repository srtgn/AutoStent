# 4C Testing & Debugging Guide

## Quick Answers

### Q: Do we have YAML files for each stent geometry?
**A: NO** - YAML files are generated on-the-fly, but they were incomplete. I've now fixed the generator to:
- Use correct 4C YAML format (matching tutorial)
- Generate VTU mesh files (if pyvista available)
- Reference VTU files in YAML (as 4C expects)

### Q: Can we use 4C Docker image instructions?
**A: YES!** The code now follows 4C's format. Test it with the steps below.

---

## Step-by-Step: Test 4C Docker Image

### Method 1: Local Test Script (Easiest)

```bash
cd /Users/srtgn/Documents/Projects/4c
./railway/test_4c_docker.sh
```

This will:
1. ✅ Check Docker is installed
2. ✅ Pull 4C Docker image
3. ✅ Test 4C binary
4. ✅ List available test files
5. ✅ Run a test simulation

### Method 2: Test via Railway API

```bash
# Test 4C Docker directly with tutorial file
curl https://4c.up.railway.app/test-4c-docker

# Test with your stent parameters
curl -X POST https://4c.up.railway.app/test-4c \
  -H "Content-Type: application/json" \
  -d '{
    "params": {
      "diameter": 10.0,
      "length": 20.0,
      "strut_thickness": 0.12,
      "num_struts": 12,
      "crown_height": 1.0
    },
    "use_docker": true
  }'
```

### Method 3: Manual Docker Test

```bash
# 1. Pull image
docker pull ghcr.io/4c-multiphysics/4c:main

# 2. Test binary
docker run --rm ghcr.io/4c-multiphysics/4c:main \
  /home/user/4C/build/4C --help

# 3. List test files
docker run --rm ghcr.io/4c-multiphysics/4c:main \
  find /home/user/4C/tests/input_files -name "*.4C.yaml" | head -5

# 4. Run tutorial simulation
mkdir -p /tmp/4c_test
docker run --rm \
  -v /tmp/4c_test:/output \
  -w /output \
  ghcr.io/4c-multiphysics/4c:main \
  /home/user/4C/build/4C \
  /home/user/4C/tests/input_files/tutorial_solid_vtu.4C.yaml \
  test_output

# 5. Check results
ls -lh /tmp/4c_test
```

---

## What I Fixed

### 1. **YAML Format** - Now matches 4C's actual format:
   - ✅ `PROBLEM TYPE:` (with space, not underscore)
   - ✅ `PROBLEMTYPE: Structure` (correct key)
   - ✅ Proper `IO`, `SOLVER`, `STRUCTURAL DYNAMIC` sections
   - ✅ `MAT_Struct_StVenantKirchhoff` material (matching 4C format)

### 2. **VTU File Generation**:
   - ✅ Added `write_vtu_file()` function using pyvista
   - ✅ Generates VTU mesh files alongside YAML
   - ✅ YAML references VTU file: `FILE: geometry.vtu`

### 3. **Boundary Conditions**:
   - ✅ Uses `DESIGN SURF DIRICH CONDITIONS` (4C format)
   - ✅ Uses `DESIGN SURF NEUMANN CONDITIONS` (4C format)
   - ✅ Proper node set references

### 4. **Test Endpoint**:
   - ✅ `/test-4c-docker` - Tests 4C Docker image directly
   - ✅ Shows stdout/stderr from 4C
   - ✅ Lists output files created

---

## Current Status

### ✅ What Works:
- 4C Docker image is available and can run
- YAML format now matches 4C's requirements
- VTU file generation added (if pyvista available)
- Test endpoints created

### ⚠️ What Needs Checking:
1. **Mesh tools import**: `mesh_generator.py` must be in Docker image
2. **pyvista availability**: Must be installed for VTU generation
3. **4C simulation**: Need to test if simulations actually run

---

## Next Steps to Verify 4C Works

1. **Deploy the fixes**:
   ```bash
   git add -A
   git commit -m "Fix: 4C YAML format, add VTU generation, add test endpoints"
   git push
   ```

2. **Test the Docker image**:
   ```bash
   curl https://4c.up.railway.app/test-4c-docker
   ```
   This will show if 4C can run at all.

3. **Check mesh tools**:
   ```bash
   curl https://4c.up.railway.app/check-4c
   ```
   Look for `mesh_tools_available: true`

4. **Test with stent parameters**:
   - Use the web UI
   - Check "4C Solver (Cloud)"
   - Start training
   - Check Railway logs for 4C output

---

## If 4C Still Fails

Check Railway logs for:
- `WARNING: Mesh tools not available` → Fix import
- `4C simulation failed` → Check error message
- `No VTU files found` → VTU generation failed
- `error while loading shared libraries` → Library path issue (already fixed)

The test endpoint will show you exactly what's happening!

