from pathlib import Path
import sys
import shutil

from setuptools import Extension, setup

try:
    import pybind11
except ImportError as exc:
    raise SystemExit(
        "pybind11 is required to build orbbec_cpp_camera. "
        "Install it with: python -m pip install pybind11"
    ) from exc


ROOT = Path(__file__).resolve().parent
SMART_CARE_ROOT = ROOT.parent
SDK_ROOT = SMART_CARE_ROOT / "third_party" / "OrbbecSDK_C_C++_v1.5.7_win_x64_release" / "SDK"
SDK_INCLUDE = SDK_ROOT / "include"
SDK_LIB = SDK_ROOT / "lib"


class BuildExtWithOrbbecDll:
    def run(self):
        super().run()
        self.copy_outputs_to_smart_care()

    def copy_outputs_to_smart_care(self):
        dll = SDK_LIB / "OrbbecSDK.dll"
        for ext in self.extensions:
            output_path = Path(self.get_ext_fullpath(ext.name)).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            target_ext = ROOT / output_path.name
            if output_path != target_ext:
                shutil.copy2(output_path, target_ext)
            if dll.exists():
                shutil.copy2(dll, output_path.parent / dll.name)
                shutil.copy2(dll, ROOT / dll.name)


try:
    from setuptools.command.build_ext import build_ext as _build_ext

    class build_ext(BuildExtWithOrbbecDll, _build_ext):
        pass

except ImportError:
    build_ext = None


setup(
    name="orbbec_cpp_camera",
    version="0.1.0",
    ext_modules=[
        Extension(
            "orbbec_cpp_camera",
            [str(ROOT / "orbbec_cpp_camera.cpp")],
            include_dirs=[str(SDK_INCLUDE), pybind11.get_include()],
            library_dirs=[str(SDK_LIB)],
            libraries=["OrbbecSDK"],
            language="c++",
            extra_compile_args=["/std:c++17", "/EHsc"] if sys.platform == "win32" else ["-std=c++17"],
        )
    ],
    cmdclass={"build_ext": build_ext} if build_ext else {},
)
