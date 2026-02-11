# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Source map generator for Allo HLS projects.

Parses generated kernel.cpp and (optionally) the original Python source
to produce a JSON source map that links RTL module names back to Allo
kernel definitions, loop labels, and source locations.

The source map enables downstream tools (e.g. LightningSim / OmniSim)
to map simulation results back to the original Allo code for
bottleneck analysis and visualization.
"""

import ast
import inspect
import json
import os
import re
import textwrap
from typing import Any, Optional


# ---------------------------------------------------------------------------
# kernel.cpp parser
# ---------------------------------------------------------------------------

# Matches:  void func_name(
_FUNC_DEF_RE = re.compile(r"^void\s+(\w+)\s*\(")

# Matches:  l_S_fa_0_fa: for (int fa = 0; fa < 16; fa++)
_LOOP_LABEL_RE = re.compile(
    r"^\s*(l_\w+):\s*for\s*\(\s*int\s+(\w+)\s*=\s*(\w+)\s*;\s*\w+\s*[<>=!]+\s*(\w+)\s*;"
)

# Matches:  for (int v4 = 0; v4 < v3; v4 += 1) {  // unlabeled variable-bound loop
_UNLABELED_FOR_RE = re.compile(
    r"^\s*for\s*\(\s*int\s+(\w+)\s*=\s*(\w+)\s*;\s*\w+\s*[<>=!]+\s*(\w+)\s*;"
)

# Matches:  #pragma HLS pipeline II=1
_PIPELINE_RE = re.compile(r"#pragma\s+HLS\s+pipeline\s+II\s*=\s*(\d+)")

# Matches:  #pragma HLS interface m_axi port=v382 ...
_MAXI_RE = re.compile(r"#pragma\s+HLS\s+interface\s+m_axi\s+port=(\w+)")

# Matches function calls:  feature_preprocess(v381, buf1, processed);
_CALL_RE = re.compile(r"^\s+(\w+)\(([^)]*)\)\s*;")

# Matches line-source comments:  // L519
_LINE_COMMENT_RE = re.compile(r"//\s*L(\d+)\s*$")


def parse_kernel_cpp(code: str) -> dict:
    """Parse generated kernel.cpp and extract structural information.

    Returns a dict with:
      - functions: {name: {params, loops, calls, is_top, line_start, line_end, ...}}
      - call_graph: {caller: [callee, ...]}
    """
    functions: dict[str, dict[str, Any]] = {}
    current_func: Optional[str] = None
    current_func_start: int = 0
    brace_depth = 0
    in_func_signature = False
    param_lines: list[str] = []

    lines = code.split("\n")
    all_func_names: set[str] = set()

    # First pass: collect all function names
    for line in lines:
        m = _FUNC_DEF_RE.match(line)
        if m:
            all_func_names.add(m.group(1))

    # Second pass: detailed parsing
    for line_no, line in enumerate(lines, start=1):
        # Detect function definition start
        m = _FUNC_DEF_RE.match(line)
        if m and current_func is None:
            fname = m.group(1)
            current_func = fname
            current_func_start = line_no
            brace_depth = 0
            in_func_signature = True
            param_lines = []
            functions[fname] = {
                "parameters": [],
                "loops": {},
                "calls": [],
                "is_top": False,
                "auto_generated": False,
                "type": "kernel",
                "hls_line_start": line_no,
                "hls_line_end": None,
                "m_axi_ports": [],
            }
            # Check if the opening brace is on this line
            if "{" in line:
                brace_depth += line.count("{") - line.count("}")
                in_func_signature = False
            continue

        if in_func_signature:
            if "{" in line:
                in_func_signature = False
                brace_depth += line.count("{") - line.count("}")
            else:
                param_lines.append(line.strip())
            continue

        if current_func is None:
            continue

        # Track brace depth
        brace_depth += line.count("{") - line.count("}")

        # Detect end of function
        if brace_depth <= 0:
            functions[current_func]["hls_line_end"] = line_no
            current_func = None
            brace_depth = 0
            continue

        finfo = functions[current_func]

        # Detect m_axi interface (identifies top function)
        m = _MAXI_RE.search(line)
        if m:
            finfo["is_top"] = True
            finfo["type"] = "top"
            finfo["m_axi_ports"].append(m.group(1))

        # Detect labeled loops
        m = _LOOP_LABEL_RE.search(line)
        if m:
            label, var, init, bound = m.groups()
            loop_info: dict[str, Any] = {
                "variable": var,
                "init": init,
                "bound": bound,
                "pipelined": False,
                "ii": None,
                "hls_line": line_no,
            }
            finfo["loops"][label] = loop_info
            # Check next line for pipeline pragma
            if line_no < len(lines):
                next_line = lines[line_no]  # 0-indexed, so lines[line_no] is next
                pm = _PIPELINE_RE.search(next_line)
                if pm:
                    loop_info["pipelined"] = True
                    loop_info["ii"] = int(pm.group(1))
            continue

        # Detect unlabeled for-loops (variable-bound loops generate VITIS_LOOP names)
        # Only match if we didn't already match a labeled loop on this line
        m = _UNLABELED_FOR_RE.search(line)
        if m and not _LOOP_LABEL_RE.search(line):
            var, init, bound = m.groups()
            # Record the kernel.cpp line number, used to resolve VITIS_LOOP_{line}_{depth}
            if "unlabeled_loops" not in finfo:
                finfo["unlabeled_loops"] = {}
            finfo["unlabeled_loops"][str(line_no)] = {
                "variable": var,
                "init": init,
                "bound": bound,
                "hls_line": line_no,
            }

        # Detect function calls (only within top function or other functions)
        m = _CALL_RE.match(line)
        if m:
            callee = m.group(1)
            if callee in all_func_names:
                finfo["calls"].append(callee)

    # Classify auto-generated load/store functions
    for fname, finfo in functions.items():
        if fname.startswith("load_buf"):
            finfo["auto_generated"] = True
            finfo["type"] = "load"
        elif fname.startswith("store_res"):
            finfo["auto_generated"] = True
            finfo["type"] = "store"

    # Build call graph
    call_graph = {}
    for fname, finfo in functions.items():
        if finfo["calls"]:
            call_graph[fname] = finfo["calls"]

    return {
        "functions": functions,
        "call_graph": call_graph,
    }


# ---------------------------------------------------------------------------
# Python source parser
# ---------------------------------------------------------------------------


def parse_allo_source(source_path: str) -> dict:
    """Parse an Allo Python source file to extract function and loop locations.

    Returns:
      {function_name: {
          "source_line_start": int,
          "source_line_end": int,
          "loops": [{variable, line, ...}, ...],
          "parameters": [name, ...],
      }}
    """
    if not os.path.isfile(source_path):
        return {}

    with open(source_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    result: dict[str, dict[str, Any]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue

        func_name = node.name
        func_start = node.lineno
        func_end = node.end_lineno if hasattr(node, "end_lineno") else func_start

        # Extract parameter names (skip type annotations for now)
        params = [arg.arg for arg in node.args.args]

        # Find all for-loops in this function
        loops = []
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                # Get loop variable name(s)
                if isinstance(child.target, ast.Name):
                    loop_var = child.target.id
                elif isinstance(child.target, ast.Tuple):
                    loop_var = ", ".join(
                        e.id for e in child.target.elts if isinstance(e, ast.Name)
                    )
                else:
                    continue

                # Try to extract the range bound
                bound = None
                if isinstance(child.iter, ast.Call):
                    func = child.iter.func
                    func_name_str = ""
                    if isinstance(func, ast.Name):
                        func_name_str = func.id
                    elif isinstance(func, ast.Attribute):
                        func_name_str = func.attr

                    if func_name_str == "range" and child.iter.args:
                        last_arg = child.iter.args[-1]
                        if isinstance(last_arg, ast.Constant):
                            bound = str(last_arg.value)
                        elif isinstance(last_arg, ast.Name):
                            bound = last_arg.id

                loops.append(
                    {
                        "variable": loop_var,
                        "line": child.lineno,
                        "end_line": child.end_lineno
                        if hasattr(child, "end_lineno")
                        else child.lineno,
                        "bound": bound,
                    }
                )

        result[func_name] = {
            "source_line_start": func_start,
            "source_line_end": func_end,
            "parameters": params,
            "loops": loops,
        }

    return result


# ---------------------------------------------------------------------------
# Matching: link HLS names to Python source
# ---------------------------------------------------------------------------


def _extract_loop_var_from_label(label: str) -> tuple[Optional[str], Optional[str]]:
    """Extract the loop variable name from an Allo-generated loop label.

    Allo labels follow the pattern: l_S_{var}_{index}_{var}
    e.g., l_S_fa_0_fa -> ('fa', 'fa')
          l_S_t_2_t   -> ('t', 't')

    When Allo renames to avoid collisions: l_S_ci_0_ci1 -> ('ci1', 'ci')
    The prefix is the original variable name, suffix may have a numeric suffix.

    For load/store labels: l_S_load_buf1_load_buf1_l_0 -> (None, None)

    Returns:
        (hls_var, original_var) where original_var has trailing digits stripped
        if the prefix matches the suffix sans trailing digits.
    """
    # Pattern: l_S_{name}_{digit}_{name}  where suffix may have extra digits
    m = re.match(r"^l_S_(.+?)_(\d+)_(.+)$", label)
    if m:
        prefix = m.group(1)
        suffix = m.group(3)
        if prefix == suffix:
            return suffix, prefix
        # Check if suffix is prefix + trailing digit(s)
        # e.g., prefix='ci', suffix='ci1' -> original was 'ci'
        stripped = re.sub(r"\d+$", "", suffix)
        if stripped == prefix:
            return suffix, prefix
    return None, None


def match_loops_to_source(
    hls_func_name: str,
    hls_loops: dict[str, dict],
    py_func_info: Optional[dict],
) -> dict[str, dict]:
    """Match HLS loop labels to Python source loop locations.

    Returns updated loop info with source_line and source_variable fields.
    """
    if py_func_info is None:
        return hls_loops

    py_loops = py_func_info.get("loops", [])

    for label, loop_info in hls_loops.items():
        hls_var = loop_info.get("variable", "")
        hls_label_var, original_var = _extract_loop_var_from_label(label)

        # Try to match by variable name
        # Priority: original_var (with trailing digits stripped) > hls_var from label
        match_vars = []
        if original_var:
            match_vars.append(original_var)
        if hls_label_var and hls_label_var != original_var:
            match_vars.append(hls_label_var)
        if hls_var and hls_var not in match_vars:
            match_vars.append(hls_var)

        for py_loop in py_loops:
            py_var = py_loop["variable"]
            # Check if any candidate variable name matches
            # Handle both exact match and comma-separated (grid) loops
            py_vars = [v.strip() for v in py_var.split(",")]
            matched = any(mv in py_vars for mv in match_vars)
            if matched:
                loop_info["source_line"] = py_loop["line"]
                loop_info["source_end_line"] = py_loop.get("end_line")
                loop_info["source_variable"] = py_var
                break

    return hls_loops


# ---------------------------------------------------------------------------
# RTL name prediction
# ---------------------------------------------------------------------------


def predict_rtl_names(functions: dict) -> dict:
    """Predict the RTL module names that Vitis HLS will generate.

    Vitis HLS naming conventions:
      - Top-level function name is preserved
      - Sub-functions become: {func_name}  (dots replaced with underscores
        in instance names, e.g. load_buf1.1 -> load_buf1_1)
      - Labeled pipelined loops: {func_name}_Pipeline_{loop_label}
      - Unlabeled loops: {func_name}_Pipeline_VITIS_LOOP_{cpp_line}_{depth}
      - Nested unlabeled: {func}_Pipeline_VITIS_LOOP_{line1}_{d1}_VITIS_LOOP_{line2}_{d2}
      - When a function is called, the instance may get a .N suffix
        (e.g. load_buf1.1) but the module name stays load_buf1

    Returns: {predicted_rtl_name: {parent_function, loop_label (if any), ...}}
    """
    rtl_map: dict[str, dict[str, Any]] = {}

    for fname, finfo in functions.items():
        # The function itself appears as an RTL module
        rtl_map[fname] = {
            "parent_function": fname,
            "type": finfo["type"],
            "is_pipeline": False,
        }

        # Each labeled loop may become a Pipeline sub-module.
        # Vitis HLS can extract loops into Pipeline sub-modules even when
        # they are not explicitly pipelined with #pragma HLS pipeline.
        for label, loop_info in finfo.get("loops", {}).items():
            rtl_name = f"{fname}_Pipeline_{label}"
            rtl_map[rtl_name] = {
                "parent_function": fname,
                "type": "pipeline",
                "is_pipeline": True,
                "loop_label": label,
                "loop_variable": loop_info.get("variable"),
                "pipelined": loop_info.get("pipelined", False),
                "ii": loop_info.get("ii"),
            }
            # Copy source info if present
            if "source_line" in loop_info:
                rtl_map[rtl_name]["source_line"] = loop_info["source_line"]

    return rtl_map


def build_vitis_loop_index(functions: dict) -> dict:
    """Build an index of kernel.cpp line numbers for VITIS_LOOP resolution.

    Vitis HLS names unlabeled loops as VITIS_LOOP_{cpp_line}_{depth}.
    This index maps cpp_line -> {parent_function, variable, ...} so that
    downstream tools can resolve VITIS_LOOP names at lookup time.

    Returns: {cpp_line_str: {parent_function, variable, ...}}
    """
    index: dict[str, dict[str, Any]] = {}

    for fname, finfo in functions.items():
        for line_str, linfo in finfo.get("unlabeled_loops", {}).items():
            entry: dict[str, Any] = {
                "parent_function": fname,
                "variable": linfo["variable"],
                "bound": linfo["bound"],
                "hls_line": linfo["hls_line"],
            }
            if "source_line" in linfo:
                entry["source_line"] = linfo["source_line"]
            index[line_str] = entry

    return index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_source_map(
    kernel_cpp_code: str,
    top_func_name: str,
    source_file: Optional[str] = None,
) -> dict:
    """Generate a source map linking HLS/RTL names to Allo source.

    Args:
        kernel_cpp_code: Contents of the generated kernel.cpp
        top_func_name: Name of the top-level HLS function
        source_file: Optional path to the Allo Python source file

    Returns:
        A dict suitable for JSON serialization containing the full source map.
    """
    # Step 1: Parse kernel.cpp
    hls_info = parse_kernel_cpp(kernel_cpp_code)
    functions = hls_info["functions"]
    call_graph = hls_info["call_graph"]

    # Step 2: Parse Python source (if available)
    py_info: dict[str, dict] = {}
    if source_file and os.path.isfile(source_file):
        py_info = parse_allo_source(source_file)

    # Step 3: Match loops to source
    for fname, finfo in functions.items():
        py_func = py_info.get(fname)
        if py_func:
            finfo["source_file"] = source_file
            finfo["source_line_start"] = py_func["source_line_start"]
            finfo["source_line_end"] = py_func["source_line_end"]
            finfo["source_parameters"] = py_func["parameters"]

        finfo["loops"] = match_loops_to_source(fname, finfo["loops"], py_func)

    # Step 4: Predict RTL names
    rtl_name_map = predict_rtl_names(functions)

    # Step 5: Build VITIS_LOOP index for unlabeled loops
    vitis_loop_index = build_vitis_loop_index(functions)

    # Step 6: Build the source map
    # Clean up internal fields not needed in output
    clean_functions = {}
    for fname, finfo in functions.items():
        clean_func: dict[str, Any] = {
            "type": finfo["type"],
            "auto_generated": finfo.get("auto_generated", False),
        }
        if finfo.get("source_file"):
            clean_func["source_file"] = finfo["source_file"]
        if finfo.get("source_line_start") is not None:
            clean_func["source_line_start"] = finfo["source_line_start"]
            clean_func["source_line_end"] = finfo["source_line_end"]
        if finfo.get("source_parameters"):
            clean_func["source_parameters"] = finfo["source_parameters"]

        # Include labeled loop info
        if finfo["loops"]:
            clean_loops = {}
            for label, linfo in finfo["loops"].items():
                clean_loop: dict[str, Any] = {
                    "variable": linfo["variable"],
                    "bound": linfo["bound"],
                    "pipelined": linfo["pipelined"],
                }
                if linfo.get("ii") is not None:
                    clean_loop["ii"] = linfo["ii"]
                if linfo.get("source_line") is not None:
                    clean_loop["source_line"] = linfo["source_line"]
                if linfo.get("source_end_line") is not None:
                    clean_loop["source_end_line"] = linfo["source_end_line"]
                if linfo.get("source_variable") is not None:
                    clean_loop["source_variable"] = linfo["source_variable"]
                clean_loops[label] = clean_loop
            clean_func["loops"] = clean_loops

        # Include unlabeled loop info (for VITIS_LOOP resolution)
        unlabeled = finfo.get("unlabeled_loops", {})
        if unlabeled:
            clean_unlabeled = {}
            for line_str, linfo in unlabeled.items():
                clean_ul: dict[str, Any] = {
                    "variable": linfo["variable"],
                    "bound": linfo["bound"],
                    "hls_line": linfo["hls_line"],
                }
                if "source_line" in linfo:
                    clean_ul["source_line"] = linfo["source_line"]
                clean_unlabeled[line_str] = clean_ul
            clean_func["unlabeled_loops"] = clean_unlabeled

        clean_functions[fname] = clean_func

    source_map = {
        "version": 1,
        "top_function": top_func_name,
        "functions": clean_functions,
        "call_graph": call_graph,
        "rtl_name_map": rtl_name_map,
        "vitis_loop_index": vitis_loop_index,
    }
    if source_file:
        source_map["source_file"] = os.path.abspath(source_file)

    return source_map


def write_source_map(source_map: dict, output_path: str) -> str:
    """Write source map to a JSON file.

    Args:
        source_map: The source map dict from generate_source_map()
        output_path: Path to write the JSON file

    Returns:
        The path written to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(source_map, f, indent=2)
    return output_path


def generate_source_map_for_project(
    project_dir: str,
    top_func_name: str,
    source_file: Optional[str] = None,
) -> Optional[str]:
    """Generate a source map for an existing HLS project directory.

    Reads kernel.cpp from the project directory and writes source_map.json
    alongside it.

    Args:
        project_dir: Path to the .prj directory
        top_func_name: Name of the top-level HLS function
        source_file: Optional path to the Allo Python source file

    Returns:
        Path to the written source_map.json, or None on failure.
    """
    kernel_path = os.path.join(project_dir, "kernel.cpp")
    if not os.path.isfile(kernel_path):
        return None

    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    source_map = generate_source_map(kernel_code, top_func_name, source_file)
    output_path = os.path.join(project_dir, "source_map.json")
    return write_source_map(source_map, output_path)
