import argparse
import runpy


MODULES = {
    "static": "fall.static",
    "dynamic": "fall.dynamic",
    "multimodal": "fall.multimodal",
}


def main():
    parser = argparse.ArgumentParser(description="SmartCare fall detection runner.")
    parser.add_argument("mode", nargs="?", choices=tuple(MODULES))
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return
    runpy.run_module(MODULES[args.mode], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
